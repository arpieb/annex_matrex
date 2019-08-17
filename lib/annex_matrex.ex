defmodule AnnexMatrex do
  @moduledoc """
  Documentation for AnnexMatrex.
  """
  alias Annex.{
    Layer.Dense,
    LayerConfig
  }

  alias AnnexMatrex.Matrix

  @spec dense(pos_integer, pos_integer, Keyword.t()) :: LayerConfig.t()
  def dense(rows, columns, opts \\ []) do
    dense_opts = [rows: rows, columns: columns, data_type: Matrix]
    LayerConfig.build(Dense, dense_opts ++ opts)
  end
end
