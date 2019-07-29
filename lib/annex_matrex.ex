defmodule AnnexMatrex do
  @moduledoc """
  Documentation for AnnexMatrex.
  """
  alias Annex.Layer.{Dense}
  alias AnnexMatrex.Matrix

  @spec dense(pos_integer, pos_integer, any) :: Dense.t()
  def dense(rows, columns, _opts \\ []) do
    %Dense{} = dense = Annex.dense(rows, columns)

    %Dense{
      dense
      | data_type: Matrix
    }
  end
end
