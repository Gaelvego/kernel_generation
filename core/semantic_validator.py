from typing import List, Union, Any
from core.schemas import MlirResponse, ScfForLoop, ScfIf, ScfYield

class SemanticValidator:
    @staticmethod
    def validate(response: MlirResponse) -> List[str]:
        errors = []
        # Inicializamos el scope con los argumentos de la función
        initial_scope = {arg.name: arg.type for arg in response.code.arguments}
        SemanticValidator._walk_operations(response.code.operations, errors, initial_scope)
        return errors

    @staticmethod
    def _walk_operations(ops: List[Any], errors: List[str], current_scope: dict):
        for op in ops:
            opcode = getattr(op, "opcode", None)
            
            # --- 1. VALIDACIÓN DE OPERANDOS INVENTADOS ---
            operands = getattr(op, "operands", [])
            for operand in operands:
                if isinstance(operand, str) and operand.startswith("%"):
                    if operand not in current_scope:
                        errors.append(f"[SEMANTIC ERROR] Variable '{operand}' is used in '{opcode}' but was never defined in this scope.")

            # --- 2. VALIDACIÓN ESPECÍFICA POR TIPO DE OPERACIÓN ---
            if isinstance(op, ScfForLoop):
                if len(op.results) != len(op.iter_args):
                    errors.append(f"[SEMANTIC ERROR] scf.for loop defines {len(op.iter_args)} iter_args but returns {len(op.results)} results. They must match exactly.")
                
                # Validar operandos iniciales de iter_args
                for arg_val in op.iter_args.values():
                    if isinstance(arg_val, str) and arg_val.startswith("%") and arg_val not in current_scope:
                        errors.append(f"[SEMANTIC ERROR] iter_arg initial value '{arg_val}' is undefined.")

                yields = [o for o in op.body if getattr(o, "opcode", None) == "scf.yield"]
                if not yields:
                    errors.append(f"[SEMANTIC ERROR] scf.for loop body is missing an scf.yield operation.")
                else:
                    last_op = op.body[-1]
                    if getattr(last_op, "opcode", None) != "scf.yield":
                        errors.append(f"[SEMANTIC ERROR] The LAST operation in an scf.for body MUST be scf.yield.")
                    for y in yields:
                        if len(y.operands) != len(op.iter_args):
                            errors.append(f"[SEMANTIC ERROR] scf.for body scf.yield returns {len(y.operands)} values, but loop iter_args expects {len(op.iter_args)} values.")
                
                if op.loop_var in current_scope:
                    errors.append(f"[SEMANTIC ERROR] Loop variable '{op.loop_var}' shadows an existing register.")
                
                new_scope = current_scope.copy()
                new_scope[op.loop_var] = "index"  # Registramos la variable del loop
                for arg in op.iter_args.keys():
                    new_scope[arg] = "unknown" # Registramos los iter_args locales
                
                SemanticValidator._walk_operations(op.body, errors, new_scope)
                
                for res in op.results:
                    current_scope[res] = "unknown" # Registramos los resultados globales
                    
            elif isinstance(op, ScfIf):
                # Checar condición
                if op.condition not in current_scope:
                    errors.append(f"[SEMANTIC ERROR] Condition '{op.condition}' for scf.if was never defined.")

                # Checar yields en Then
                yields_then = [o for o in op.then_body if getattr(o, "opcode", None) == "scf.yield"]
                if not yields_then and len(op.results) > 0:
                     errors.append(f"[SEMANTIC ERROR] scf.if 'then' block must have an scf.yield because the operation expects {len(op.results)} results.")
                if yields_then:
                    for y in yields_then:
                        if len(y.operands) != len(op.results):
                            errors.append(f"[SEMANTIC ERROR] scf.if 'then' block yields {len(y.operands)} values, but expects {len(op.results)}.")
                
                # Checar yields en Else
                if op.else_body:
                    yields_else = [o for o in op.else_body if getattr(o, "opcode", None) == "scf.yield"]
                    if not yields_else and len(op.results) > 0:
                         errors.append(f"[SEMANTIC ERROR] scf.if 'else' block must have an scf.yield because the operation expects {len(op.results)} results.")
                    if yields_else:
                        for y in yields_else:
                            if len(y.operands) != len(op.results):
                                errors.append(f"[SEMANTIC ERROR] scf.if 'else' block yields {len(y.operands)} values, but expects {len(op.results)}.")
                
                SemanticValidator._walk_operations(op.then_body, errors, current_scope.copy())
                if op.else_body:
                    SemanticValidator._walk_operations(op.else_body, errors, current_scope.copy())
                
                for res in op.results:
                    current_scope[res] = "unknown"
                    
            elif opcode == "scf.yield":
                pass
                
            else:
                # --- 3. VALIDACIÓN DE TENSORES (Evita el crash de C++ del Intento 2) ---
                if opcode == "tensor.extract":
                    target_tensor = operands[0] if operands else None
                    if target_tensor in current_scope:
                        tensor_type = str(current_scope[target_tensor])
                        # Extraer dimensiones (ej: tensor<128x128xf32> -> 2 dimensiones)
                        if "x" in tensor_type and "tensor" in tensor_type:
                            dimensions = tensor_type.count("x")
                            expected_operands = dimensions + 1 # 1 para el tensor + N índices
                            if len(operands) != expected_operands:
                                errors.append(f"[SEMANTIC ERROR] 'tensor.extract' on {tensor_type} requires {expected_operands} operands (tensor + {dimensions} indices). You provided {len(operands)}.")

                # Registrar el resultado en el scope
                result = getattr(op, "result", None)
                out_type = getattr(op, "out_type", "unknown")
                if result and result != "none":
                    if result in current_scope:
                        errors.append(f"[SEMANTIC ERROR] Register '{result}' is assigned multiple times (shadowing).")
                    current_scope[result] = out_type

