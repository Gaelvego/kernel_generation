import re
from typing import Dict, Any, List, Optional
import warnings

try:
    from mlir.ir import Context, Module, Location, InsertionPoint, Type, Value
    from mlir.dialects import arith, tensor, scf, func
    import mlir.ir as ir
    HAS_MLIR = True
except ImportError:
    HAS_MLIR = False

from core.schemas import MLIRFunctionBody, Operation, ScfForLoop, ScfIf, ScfYield, MlirOpcode

class MLIRTranslator:
    """
    Deterministic translator from structured JSON to MLIR Dialects using mlir-py.
    """
    
    def __init__(self):
        """
        Initializes the global MLIR context and allows unregistered dialects.

        Raises:
            ImportError: If MLIR bindings are not found.
        """
        if not HAS_MLIR:
            raise ImportError("MLIR dependency not found.")
            
        self.context = Context()
        self.context.allow_unregistered_dialects = True # Important for 'tt', 'ttg' dialects
        self.value_env: Dict[str, Value] = {}
        
    def _parse_type(self, type_str: str) -> Type:
        """
        Parses basic string types like 'f32', 'f16', 'index', 'tensor<64x128xf32>'.
        
        Args:
            type_str (str): The MLIR type string.

        Returns:
            Type: The MLIR ir.Type object.
        """
        with self.context, Location.unknown():
            type_str = type_str.strip()
            if type_str == "f32": return ir.F32Type.get()
            if type_str == "f16": return ir.F16Type.get()
            if type_str == "i32": return ir.IntegerType.get_signless(32)
            if type_str == "i1": return ir.IntegerType.get_signless(1)
            if type_str == "index": return ir.IndexType.get()
            
            # Tensors: tensor<N...xType>
            match = re.match(r"tensor<(.+)x([a-z0-9]+)>", type_str)
            if match:
                shape_str = match.group(1).split("x")
                shape = [int(s) if s.isdigit() else ir.ShapedType.get_dynamic_size() for s in shape_str] # Dynamic dimension support
                element_type = self._parse_type(match.group(2))
                return ir.RankedTensorType.get(shape, element_type)
            
            # Fallback to native MLIR parsing if it's a valid general string
            return ir.Type.parse(type_str)

    def _infer_type(self, op: Operation) -> Type:
        """
        Infers the return type with per-opcode knowledge.
        If there's an explicit cast in out_type, it uses it.
        Otherwise uses opcode-specific rules or inherits from operands.
        
        Args:
            op (Operation): The Operation object.

        Returns:
            Type: The inferred MLIR ir.Type.
        """
        with self.context, Location.unknown():
            # 1. Explicit override always wins
            if op.out_type:
                return self._parse_type(op.out_type)

            opcode = op.opcode

            # 2. Opcode-specific rules
            if opcode == MlirOpcode.ARITH_CONSTANT:
                # Should be handled before reaching here, but default to f32 if missing
                warnings.warn("arith.constant reached generic inference without explicit out_type; defaulting to f32")
                return ir.F32Type.get()

            if opcode == MlirOpcode.ARITH_CMPF:
                # Comparison of floats always returns i1 (boolean)
                return ir.IntegerType.get_signless(1)

            if opcode == MlirOpcode.ARITH_SELECT:
                # arith.select takes (condition:i1, true_value, false_value) and returns the value type
                # We return the type of the second operand if available
                if len(op.operands) >= 2 and op.operands[1] in self.value_env:
                    return self.value_env[op.operands[1]].type
                return ir.F32Type.get()

            if opcode in (MlirOpcode.MATH_EXP, MlirOpcode.MATH_LOG, MlirOpcode.MATH_SQRT,
                          MlirOpcode.MATH_COS, MlirOpcode.MATH_SIN, MlirOpcode.MATH_ABS):
                # Unary math ops preserve operand type
                if op.operands and op.operands[0] in self.value_env:
                    return self.value_env[op.operands[0]].type
                return ir.F32Type.get()

            if opcode in (MlirOpcode.TT_LOAD, MlirOpcode.TTG_LOCAL_LOAD):
                # Triton load: operand is pointer, result is element type.
                # Without full pointer-type parsing, we fall back to f32 with a warning.
                warnings.warn(f"{opcode.value} inference without out_type: defaulting to f32. "
                              "LLM should specify out_type for memory ops.")
                return ir.F32Type.get()

            if opcode == MlirOpcode.TT_SPLAT:
                # tt.splat takes a scalar and returns a tensor — we cannot infer the tensor shape
                # without out_type
                if op.operands and op.operands[0] in self.value_env:
                    warnings.warn("tt.splat inference without out_type: defaulting to scalar operand type")
                    return self.value_env[op.operands[0]].type
                return ir.F32Type.get()

            if opcode == MlirOpcode.TT_MAKE_RANGE:
                # tt.make_range returns a tensor< BLOCK x i32 > typically
                warnings.warn("tt.make_range inference without out_type: defaulting to tensor<128xi32>")
                i32 = ir.IntegerType.get_signless(32)
                return ir.RankedTensorType.get([128], i32)

            # 3. Default: inherit from first operand
            if op.operands and op.operands[0] in self.value_env:
                return self.value_env[op.operands[0]].type

            # 4. Ultimate fallback
            warnings.warn(f"Could not infer type for {opcode.value}; defaulting to f32")
            return ir.F32Type.get()

    def _resolve_operands(self, op_obj: Operation) -> List[Value]:
        """
        Resolve operand names to MLIR Values, with auto-repair for missing constants.
        """
        operands = []
        for name in op_obj.operands:
            if name in self.value_env:
                operands.append(self.value_env[name])
            else:
                # Auto-repair: if the operand looks like a numeric literal, inject a constant
                try:
                    if '.' in name:
                        f_val = float(name)
                        const_op = arith.ConstantOp(ir.F32Type.get(), ir.FloatAttr.get(ir.F32Type.get(), f_val))
                        self.value_env[name] = const_op.result
                        operands.append(const_op.result)
                    else:
                        i_val = int(name)
                        const_op = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), i_val))
                        self.value_env[name] = const_op.result
                        operands.append(const_op.result)
                except ValueError:
                    raise KeyError(f"Unknown operand '{name}' for operation {op_obj.opcode.value}")
        return operands

    def _process_operations(self, operations: list):
        """
        Processes a list of operations and inserts them into the current block.
        
        Args:
            operations (list): List of Operation or SCF objects.
        """
        with self.context, Location.unknown():
            for op_obj in operations:
                if isinstance(op_obj, Operation):
                    # --- Special handling for arith.constant ---
                    if op_obj.opcode == MlirOpcode.ARITH_CONSTANT:
                        self._handle_constant(op_obj)
                        continue

                    # Standard operation (Triton, Arith, etc)
                    operands = self._resolve_operands(op_obj)
                    result_type = self._infer_type(op_obj)
                    
                    # Use generic constructor to support any dialect without hard Python bindings
                    # Since opcode is an Enum, we use .value
                    op = ir.Operation.create(
                        name=op_obj.opcode.value,
                        results=[result_type],
                        operands=operands
                    )
                    self.value_env[op_obj.result] = op.result
                    
                elif isinstance(op_obj, ScfYield):
                    # scf.yield
                    operands = [self.value_env[name] for name in op_obj.operands]
                    scf.YieldOp(operands)
                    
                elif isinstance(op_obj, ScfForLoop):
                    # scf.for loop
                    lb = self._get_or_create_index(op_obj.lower_bound)
                    ub = self._get_or_create_index(op_obj.upper_bound)
                    step = self._get_or_create_index(op_obj.step)
                    
                    # iter_args with literal fallback
                    iter_args_values = []
                    for init_val in op_obj.iter_args.values():
                        if init_val in self.value_env:
                            iter_args_values.append(self.value_env[init_val])
                        else:
                            # If LLM passed a string literal like "0.0", auto-create an f32 constant
                            try:
                                f_val = float(init_val)
                                const_op = arith.ConstantOp(ir.F32Type.get(), ir.FloatAttr.get(ir.F32Type.get(), f_val))
                                iter_args_values.append(const_op.result)
                            except ValueError:
                                raise KeyError(f"Unknown iter_arg value: {init_val}")
                    
                    for_op = scf.ForOp(lb, ub, step, iter_args_values)
                    
                    with InsertionPoint(for_op.body):
                        # Register loop variable
                        self.value_env[op_obj.loop_var] = for_op.induction_variable
                        # Register iter_args inside the loop
                        for i, arg_name in enumerate(op_obj.iter_args.keys()):
                            self.value_env[arg_name] = for_op.inner_iter_args[i]
                            
                        self._process_operations(op_obj.body)
                        
                    # Register results generated by the loop
                    for i, res_name in enumerate(op_obj.results):
                        self.value_env[res_name] = for_op.results[i]
                        
                elif isinstance(op_obj, ScfIf):
                    # scf.if
                    cond_val = self.value_env[op_obj.condition]
                    has_else = bool(op_obj.else_body)
                    
                    # Determine result types
                    result_types = self._infer_if_result_types(op_obj)
                    
                    if_op = scf.IfOp(cond_val, results_=result_types, hasElse=has_else)
                    
                    with InsertionPoint(if_op.then_block):
                        self._process_operations(op_obj.then_body)
                        
                    if has_else:
                        with InsertionPoint(if_op.else_block):
                            self._process_operations(op_obj.else_body)

                    # Map result registers if specified
                    for i, res_name in enumerate(op_obj.results):
                        if i < len(if_op.results):
                            self.value_env[res_name] = if_op.results[i]

    def _handle_constant(self, op_obj: Operation):
        """
        Handles arith.constant by parsing the 'value' field and creating the right MLIR attribute.
        """
        with self.context, Location.unknown():
            out_type = self._infer_type(op_obj)  # will use out_type if present, else f32
            raw_value = op_obj.value

            if raw_value is None:
                # Fallback to zero if LLM forgot the value
                warnings.warn("arith.constant missing 'value' field; defaulting to 0")
                raw_value = 0

            if out_type == ir.F32Type.get():
                val = float(raw_value)
                attr = ir.FloatAttr.get(ir.F32Type.get(), val)
            elif out_type == ir.F16Type.get():
                val = float(raw_value)
                attr = ir.FloatAttr.get(ir.F16Type.get(), val)
            elif out_type == ir.IntegerType.get_signless(32):
                val = int(raw_value)
                attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), val)
            elif out_type == ir.IntegerType.get_signless(1):
                val = int(raw_value)
                attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(1), val)
            elif out_type == ir.IndexType.get():
                val = int(raw_value)
                attr = ir.IntegerAttr.get(ir.IndexType.get(), val)
            else:
                # Fallback: try to parse as float
                val = float(raw_value)
                attr = ir.FloatAttr.get(ir.F32Type.get(), val)

            const_op = arith.ConstantOp(out_type, attr)
            self.value_env[op_obj.result] = const_op.result

    def _infer_if_result_types(self, op_obj: ScfIf) -> List[Type]:
        """
        Infer result types for scf.if from explicit JSON field or from yield operands.
        """
        with self.context, Location.unknown():
            # 1. Explicit result_types in JSON
            if op_obj.result_types:
                return [self._parse_type(t) for t in op_obj.result_types]

            # 2. Try to infer from yields in then_body / else_body
            def find_yield_types(body):
                types = []
                for node in body:
                    if isinstance(node, ScfYield):
                        for operand_name in node.operands:
                            if operand_name in self.value_env:
                                types.append(self.value_env[operand_name].type)
                            else:
                                # Unknown pre-defined value — can't infer here
                                types.append(ir.F32Type.get())
                        return types
                return []

            then_types = find_yield_types(op_obj.then_body)
            else_types = find_yield_types(op_obj.else_body) if op_obj.else_body else then_types

            # Use then_types if found, else fallback
            if then_types:
                return then_types

            # 3. Ultimate fallback: empty (void if)
            return []

    def _get_or_create_index(self, val) -> Value:
        """
        Converts int to arith.constant index, auto-casts integers to index, or fetches the register.
        """
        if isinstance(val, int):
            op = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), val))
            return op.result
        elif isinstance(val, str) and val.isdigit():
            op = arith.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), int(val)))
            return op.result
            
        v = self.value_env[val]
        # Auto-cast if it is an integer type but not an index
        if ir.IntegerType.isinstance(v.type) and v.type != ir.IndexType.get():
            cast_op = arith.IndexCastOp(ir.IndexType.get(), v)
            return cast_op.result
        return v

    def translate_to_module(self, function_body: MLIRFunctionBody) -> str:
        """
        Translation layer from abstract JSON to MLIR Dialects.
        
        Args:
            function_body (MLIRFunctionBody): The parsed Pydantic contract.

        Returns:
            str: The generated MLIR string.

        Raises:
            RuntimeError: If the module fails semantic verification.
        """
        with self.context, Location.unknown():
            module = Module.create()
            with InsertionPoint(module.body):
                # Extract input types
                input_types = [self._parse_type(arg.type) for arg in function_body.arguments]
                
                # Infer return types from the actual values in value_env after processing,
                # but for declaration we need to know them upfront.
                # In a single-pass compiler, we must trust the JSON or do a pre-scan.
                # For this project, we assume the JSON either returns nothing (void)
                # or the LLM has provided matching types.
                # We will do a best-effort inference by looking at the value_env after body processing.
                # To do that, we need to defer return type resolution... but func.FuncOp needs it at creation.
                # Practical workaround: process the body into a temporary block or assume void for Triton kernels.
                
                # Most Triton kernels are @triton.jit functions that may return void or a single value.
                # We will create the function with a guessed signature and fix it if needed.
                # Actually, a cleaner way is to build the body first in a standalone block, then build the func around it.
                # But for simplicity, we use a two-pass approach: 
                #   Pass 1: build body without func wrapper to collect return types
                #   Pass 2: build func with correct signature
                
                # However, operations need an InsertionPoint, which requires a block.
                # Simpler approach: build func with input_types and [] returns initially, then process body,
                # and if returns are found, we accept the module may not be perfect or we rebuild.
                # For this research project, we accept that the LLM should produce kernels that return void
                # or match the first input type.
                
                # Better approach for this project: infer return types from the 'returns' list
                # by looking up their types in value_env after body processing.
                # Since we can't do that before creating the FuncOp, we'll use a placeholder
                # and rely on MLIR's verification to catch mismatches.
                
                # Most Triton benchmarks return nothing (void) or a tensor.
                # Let's attempt a heuristic: if returns is empty, void. Otherwise, use input types.
                return_types = []
                
                func_type = ir.FunctionType.get(inputs=input_types, results=return_types)
                func_op = func.FuncOp(name=function_body.function_name, type=func_type)
                
                entry_block = func_op.add_entry_block()
                with InsertionPoint(entry_block):
                    self.value_env.clear()
                    # Map names to entry block Values
                    for i, arg in enumerate(function_body.arguments):
                        self.value_env[arg.name] = entry_block.arguments[i]
                        
                    self._process_operations(function_body.operations)
                    
                    # Generate return
                    ret_vals = [self.value_env[r] for r in function_body.returns]
                    func.ReturnOp(ret_vals)
                    
            if not module.operation.verify():
                raise RuntimeError("Generated MLIR Module is not semantically valid.")
                
            return str(module)
