defmodule AxonOnnx.Mapping do
  @moduledoc """
    Helper module used for mapping different data types
    (partially ported from: https://github.com/onnx/onnx/blob/master/onnx/helper.py)
  """

  @my_tensor_type_to_nx_size [
    {:UNDEFINED, :UNDEFINED},
    {:FLOAT, 4},
    {:UINT8, 1},
    {:INT8, 1},
    {:UINT16, 2},
    {:INT16, 2},
    {:INT32, 4},
    {:INT64, 8},
    {:STRING, :STRING},
    {:BOOL, :BOOL},
    {:FLOAT16, 2},
    {:DOUBLE, 8},
    {:UINT32, 4},
    {:UINT64, 8},
    {:COMPLEX64, :COMPLEX64},
    {:COMPLEX128, :COMPLEX128},
    {:BFLOAT16, 2}
  ]

  @my_tensor_type_atom_to_storage_type [
    {:UNDEFINED, :UNDEFINED},
    {:FLOAT, :FLOAT},
    {:UINT8, :UINT32},
    {:INT8, :INT32},
    {:UINT16, :INT32},
    {:INT16, :INT32},
    {:INT32, :INT32},
    {:INT64, :INT64},
    {:STRING, :STRING},
    {:BOOL, :INT32},
    {:FLOAT16, :UINT16},
    {:DOUBLE, :DOUBLE},
    {:UINT32, :UINT32},
    {:UINT64, :UINT64},
    {:COMPLEX64, :FLOAT},
    {:COMPLEX128, :DOUBLE},
    {:BFLOAT16, :UINT16}
  ]

  @my_storage_tensor_type_to_field [
    {:FLOAT, :float_data},
    {:INT32, :int32_data},
    {:INT64, :int64_data},
    {:UINT16, :int32_data},
    {:DOUBLE, :double_data},
    {:COMPLEX64, :float_data},
    {:COMPLEX128, :double_data},
    {:UINT32, :uint64_data},
    {:UINT64, :uint64_data},
    {:STRING, :string_data},
    {:BOOL, :int32_data}
  ]

  def tensor_type_atom_to_storage_type, do: @my_tensor_type_atom_to_storage_type
  def storage_tensor_type_to_field, do: @my_storage_tensor_type_to_field
  def tensor_type_to_nx_size, do: @my_tensor_type_to_nx_size
end
