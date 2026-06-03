from typing import List, Optional, Union, Dict
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# SCF & Triton Schema (Structured Outputs)
# ---------------------------------------------------------

class InputArgument(BaseModel):
    """
    Represents an input argument for the function.
    The LLM MUST provide the explicit type here.
    """
    name: str = Field(..., pattern=r"^%[a-zA-Z0-9_]+$", description="Register name, e.g., '%arg0'")
    type: str = Field(..., pattern=r"^(tensor<([0-9a-zA-Z]+x)+f(32|16)>|f32|f16|i32|i1|index)$", description="Exact MLIR type, e.g., 'tensor<1024xf32>'")

from enum import Enum

class MlirOpcode(str, Enum):
    # Arith Dialect
    ARITH_ADDF = "arith.addf"
    ARITH_SUBF = "arith.subf"
    ARITH_MULF = "arith.mulf"
    ARITH_DIVF = "arith.divf"
    ARITH_MAXIMUMF = "arith.maximumf"
    ARITH_MINIMUMF = "arith.minimumf"
    ARITH_CMPF = "arith.cmpf"
    ARITH_CONSTANT = "arith.constant"
    ARITH_EXTF = "arith.extf"
    ARITH_TRUNCF = "arith.truncf"
    ARITH_SELECT = "arith.select"
    
    # Math Dialect
    MATH_EXP = "math.exp"
    MATH_LOG = "math.log"
    MATH_SQRT = "math.sqrt"
    MATH_COS = "math.cos"
    MATH_SIN = "math.sin"
    MATH_ABS = "math.absf"

    # Tensor Dialect
    TENSOR_EMPTY = "tensor.empty"
    TENSOR_EXTRACT = "tensor.extract"
    TENSOR_INSERT = "tensor.insert"

    # Triton (tt) Dialect
    TT_LOAD = "tt.load"
    TT_STORE = "tt.store"
    TT_SPLAT = "tt.splat"
    TT_BROADCAST = "tt.broadcast"
    TT_EXPAND_DIMS = "tt.expand_dims"
    TT_DOT = "tt.dot"
    TT_REDUCE = "tt.reduce"
    TT_RESHAPE = "tt.reshape"
    TT_TRANS = "tt.trans"
    TT_MAKE_RANGE = "tt.make_range"
    
    # Triton GPU (ttg) Dialect
    TTG_LOCAL_ALLOC = "ttg.local_alloc"
    TTG_LOCAL_LOAD = "ttg.local_load"
    TTG_LOCAL_STORE = "ttg.local_store"
    TTG_CONVERT_LAYOUT = "ttg.convert_layout"

class Operation(BaseModel):
    """
    Standard Triton/Arith math or memory operation.
    """
    opcode: MlirOpcode = Field(..., description="Allowed MLIR/Triton operation")
    operands: List[str] = Field(..., description="Input registers (e.g., ['%0', '%1'])")
    result: str = Field(..., pattern=r"^%[a-zA-Z0-9_]+$|^none$", description="Output register (e.g., '%2') or 'none'")
    out_type: Optional[str] = Field(None, pattern=r"^(tensor<([0-9a-zA-Z]+x)+f(32|16)>|f32|f16|i32|i1|index)$", description="Specify ONLY for explicit casting (e.g., 'f16').")
    value: Optional[Union[float, int, str]] = Field(None, description="Literal value for arith.constant or similar attribute-bearing ops. Ignored for normal ops.")

class ScfYield(BaseModel):
    """
    Return operation within an SCF block.
    """
    opcode: str = Field("scf.yield", description="Must always be 'scf.yield'")
    operands: List[str] = Field(default_factory=list, description="Values yielded to the next loop iter_arg or out of the if-block")

class ScfIf(BaseModel):
    """
    If-then-else block in SCF.
    """
    opcode: str = Field("scf.if", description="Must always be 'scf.if'")
    condition: str = Field(..., description="Register containing the boolean (i1) condition")
    results: List[str] = Field(default_factory=list, description="Names of the registers where the scf.if result is stored")
    result_types: Optional[List[str]] = Field(None, description="MLIR types of the results (e.g., ['f32', 'i32']). If omitted, translator will attempt inference from yields.")
    then_body: List[Union['Operation', 'ScfForLoop', 'ScfIf', 'ScfYield']] = Field(...)
    else_body: Optional[List[Union['Operation', 'ScfForLoop', 'ScfIf', 'ScfYield']]] = Field(None)

class ScfForLoop(BaseModel):
    """
    For loop (scf.for) with support for loop-carried variables (iter_args).
    """
    opcode: str = Field("scf.for", description="Must always be 'scf.for'")
    lower_bound: Union[int, str] = Field(..., description="Lower bound (constant integer or register)")
    upper_bound: Union[int, str] = Field(..., description="Upper bound")
    step: Union[int, str] = Field(..., description="Step size")
    loop_var: str = Field(..., description="Iterator register, e.g., '%i'")
    iter_args: Dict[str, str] = Field(default_factory=dict, description="Dictionary {Iter_Arg_Name: Initial_Value_Name}")
    results: List[str] = Field(default_factory=list, description="Return registers generated after loop termination")
    body: List[Union['Operation', 'ScfForLoop', 'ScfIf', 'ScfYield']] = Field(...)

# Rebuild recursive references (required in Pydantic for recursive Unions)
ScfIf.model_rebuild()
ScfForLoop.model_rebuild()

class MLIRFunctionBody(BaseModel):
    """
    Complete function structure.
    """
    function_name: str
    arguments: List[InputArgument] = Field(..., description="Input variables WITH their types")
    operations: List[Union[Operation, ScfForLoop, ScfIf]] = Field(..., description="Body of operations")
    returns: List[str] = Field(..., description="Registers returned by the function")

class MlirResponse(BaseModel):
    """
    Final contract for the LLM response (Structured Output).
    """
    reasoning: str = Field(..., description="Step-by-step reasoning before emitting the code.")
    code: MLIRFunctionBody = Field(..., description="The program compiled to JSON")
