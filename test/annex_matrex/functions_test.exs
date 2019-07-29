defmodule AnnexMatrex.FunctionsTest do
  use ExUnit.Case
  alias AnnexMatrex.Functions

  setup do
    value = 0.4
    matrix = Matrex.fill(3, 5, value)
    {:ok, matrix: matrix, value: value}
  end

  def assert_valid_matrix(matrix, expected_value) do
    assert %Matrex{} = matrix
    assert Matrex.size(matrix) == {3, 5}

    Enum.each(matrix, fn v ->
      assert_in_delta(v, expected_value, 0.0001)
    end)
  end

  describe "apply/2" do
    test "works for {:derivative, :sigmoid}", %{matrix: matrix} do
      matrix
      |> Functions.apply({:derivative, :sigmoid})
      |> assert_valid_matrix(0.24026075)
    end

    test "works for :identity", %{matrix: matrix} do
      matrix
      |> Functions.apply(:identity)
      |> assert_valid_matrix(0.4)
    end

    test "works for {:derivative, :identity}", %{matrix: matrix} do
      matrix
      |> Functions.apply({:derivative, :identity})
      |> assert_valid_matrix(1.0)
    end

    test "works for :relu", %{matrix: matrix} do
      matrix
      |> Functions.apply(:relu)
      |> assert_valid_matrix(0.4)
    end

    test "works for {:derivative, :relu}", %{matrix: matrix} do
      matrix
      |> Functions.apply({:derivative, :relu})
      |> assert_valid_matrix(1.0)
    end

    test "works for :softmax", %{matrix: matrix} do
      matrix
      |> Functions.apply(:softmax)
      |> assert_valid_matrix(0.06666667)
    end

    test "works for :softmax_derivative", %{matrix: matrix} do
      matrix
      |> Functions.apply({:derivative, :softmax})
      |> assert_valid_matrix(0.4000000059604645)
    end

    test "works for {:derivative, :tanh}", %{matrix: matrix} do
      matrix
      |> Functions.apply({:derivative, :tanh})
      |> assert_valid_matrix(0.8556387424468994)
    end

    test "raises for an invalid function name", %{matrix: matrix} do
      assert_raise(ArgumentError, fn -> Functions.apply(matrix, :bleppppp) end)
    end
  end

  describe "relu/1" do
    test "returns identity when zero or greater" do
      Enum.each(0..24, fn i ->
        w = 1.0 * :math.sqrt(i)
        assert Functions.relu(w) == w
      end)
    end

    test "returns 0.0 for negatives" do
      Enum.each(0..24, fn i ->
        w = -1.0 * :math.sqrt(i)
        assert Functions.relu(w) == 0.0
      end)
    end
  end

  describe "relu_derivative/1" do
    test "returns 1.0 for positives" do
      Enum.each(1..24, fn i ->
        w = 1.0 * :math.sqrt(i)
        assert Functions.relu_derivative(w) == 1.0
      end)
    end

    test "returns 0.0 for 0.0" do
      assert Functions.relu_derivative(0.0) == 0.0
    end

    test "returns 0.0 for negatives" do
      Enum.each(1..24, fn i ->
        w = -1.0 * :math.sqrt(i)
        assert Functions.relu_derivative(w) == 0.0
      end)
    end
  end
end
