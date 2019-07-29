defmodule AnnexMatrex.Functions do
  @type function_name ::
          :identity
          | {:derivative, :identity}
          | :relu
          | {:derivative, :relu}
          | :softmax
          | {:derivative, :softmax}
          | {:derivative, :tanh}
          | {:derivative, :sigmoid}

  @spec apply(Matrex.t(), function_name()) :: Matrex.t()
  def apply(%Matrex{} = m, {:derivative, :sigmoid}) do
    m
    |> Matrex.apply(:sigmoid)
    |> Matrex.apply(fn fx_sigmoid -> (1.0 - fx_sigmoid) * fx_sigmoid end)
  end

  def apply(%Matrex{} = m, :identity) do
    m
  end

  def apply(%Matrex{} = m, {:derivative, :identity}) do
    {rows, columns} = Matrex.size(m)
    Matrex.ones(rows, columns)
  end

  def apply(%Matrex{} = m, :relu), do: Matrex.apply(m, &relu/1)
  def apply(%Matrex{} = m, {:derivative, :relu}), do: Matrex.apply(m, &relu_derivative/1)

  def apply(%Matrex{} = m, :softmax) do
    # get the max weight
    w_max = Matrex.max(m)

    # calculate weight - w_max for each weight
    # get the exponent
    exps =
      m
      |> Matrex.subtract(w_max)
      |> Matrex.apply(:exp)

    # get the sum
    sum = Matrex.sum(exps)

    # put the exp in a list  sum the exponents
    Matrex.divide(exps, sum)
  end

  def apply(%Matrex{} = m, {:derivative, :softmax}), do: m

  def apply(%Matrex{} = m, {:derivative, :tanh}) do
    m
    |> Matrex.apply(:tanh)
    |> Matrex.square()
    |> Matrex.multiply(-1.0)
    |> Matrex.add(1.0)
  end

  def apply(%Matrex{} = _m, func) do
    raise ArgumentError,
      message: """
      AnnexMatrex.Functions.apply/2 encountered an unmatched function: #{inspect(func)}
      """
  end

  def relu(w), do: max(0.0, w)

  def relu_derivative(w) when w > 0, do: 1.0
  def relu_derivative(_), do: 0.0
end
