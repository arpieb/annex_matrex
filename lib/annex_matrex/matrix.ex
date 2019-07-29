defmodule AnnexMatrex.Matrix do
  use Annex.Data
  use Annex.Debug, debug: true

  alias AnnexMatrex.Matrix

  @type inner :: Matrex.t()
  @type t :: %Matrix{
          inner: inner
        }

  defstruct inner: nil

  defp to_inner(%Matrix{inner: inner}), do: inner
  defp to_inner(%Matrex{} = inner), do: inner
  defp to_inner(inner) when is_float(inner), do: inner

  def build(%Matrex{} = inner) do
    %Matrix{inner: inner}
  end

  @impl Data
  @spec cast(Data.flat_data() | Matrex.t() | t(), {any, any}) :: t()
  def cast(%Matrix{} = data, {rows, columns}) do
    data
    |> to_inner()
    |> cast({rows, columns})
  end

  def cast(%Matrex{} = inner, {rows, columns}) do
    case Matrex.size(inner) do
      {^rows, ^columns} ->
        inner

      _ ->
        Matrex.reshape(inner, rows, columns)
    end
    |> build
  end

  def cast(data, {rows, columns}) when Data.is_flat_data(data) do
    case {length(data), rows, columns} do
      {size, rows, columns} when rows * columns == size ->
        Matrex.reshape(data, rows, columns)

      {same, _, same} ->
        Matrex.reshape(data, 1, columns)

      {same, same, _} ->
        Matrex.reshape(data, rows, 1)

      _ ->
        raise ArgumentError,
          message: """
          AnnexMatrex.Matrix.cast/2 encountered invalid data and shape.

          data: #{inspect(data)}
          shape: #{inspect({rows, columns})}
          """
    end
    |> build()
  end

  @native_functions [
    :exp,
    :exp2,
    :sigmoid,
    :expm1,
    :log,
    :log2,
    :sqrt,
    :cbrt,
    :ceil,
    :floor,
    :truncate,
    :round,
    :abs,
    :sin,
    :cos,
    :tan,
    :asin,
    :acos,
    :atan,
    :sinh,
    :cosh,
    :tanh,
    :asinh,
    :acosh,
    :atanh,
    :erf,
    :erfc,
    :tgamma,
    :lgamma
  ]

  @matrex_functions_2 [
    :add,
    :subtract,
    :multiply,
    :dot
  ]

  @other_functions [
    :identity,
    {:derivative, :identity},
    :relu,
    {:derivative, :relu},
    :softmax,
    {:derivative, :softmax},
    {:derivative, :tanh},
    {:derivative, :sigmoid}
  ]

  @impl Data
  @spec apply_op(t(), any(), [t() | number | inner()]) :: t()
  def apply_op(%Matrix{} = matrix, op, args) do
    inner = to_inner(matrix)

    case {op, args} do
      {_, []} when op in @other_functions ->
        AnnexMatrex.Functions.apply(inner, op)

      {_, _} when op in @native_functions ->
        Matrex.apply(inner, op)

      {_, [other]} when op in @matrex_functions_2 ->
        apply(Matrex, op, [inner, to_inner(other)])

      {:transpose, []} ->
        Matrex.transpose(inner)

      {:map, [func]} when is_function(func, 1) or is_function(func, 2) or is_function(func, 3) ->
        Matrex.apply(inner, func)
    end
    |> build
  end

  @impl Data
  @spec is_type?(any) :: boolean
  def is_type?(%Matrix{}), do: true
  def is_type?(_), do: false

  @impl Data
  @spec shape(t()) :: {pos_integer, pos_integer}
  def shape(%Matrix{} = m), do: m |> to_inner() |> Matrex.size()

  @impl Data
  @spec to_flat_list(t()) :: [float]
  def to_flat_list(%Matrix{} = m), do: m |> to_inner() |> Matrex.to_list()
end
