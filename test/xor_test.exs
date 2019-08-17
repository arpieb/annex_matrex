defmodule AnnexMatrex.SequenceXorTest do
  use ExUnit.Case, async: true

  def layers() do
    [
      AnnexMatrex.dense(8, 2),
      Annex.activation(:tanh),
      AnnexMatrex.dense(1, 8),
      Annex.activation(:sigmoid)
    ]
  end

  def sequence do
    Annex.sequence(layers())
  end

  def dataset do
    [
      {[0.0, 0.0], [0.0]},
      {[0.0, 1.0], [1.0]},
      {[1.0, 0.0], [1.0]},
      {[1.0, 1.0], [0.0]}
    ]
  end

  def train_opts do
    [
      name: "XOR operation",
      learning_rate: 0.05,
      halt_condition: {:epochs, 8_000},
      log_interval: 10_000
    ]
  end

  test "xor test" do
    seq1 = sequence()

    assert {seq2, _loss} = Annex.train(seq1, dataset(), train_opts())

    assert [zero_zero] = Annex.predict(seq2, [0.0, 0.0])
    assert [zero_one] = Annex.predict(seq2, [0.0, 1.0])
    assert [one_zero] = Annex.predict(seq2, [1.0, 0.0])
    assert [one_one] = Annex.predict(seq2, [1.0, 1.0])

    assert_in_delta(zero_one, 1.0, 0.1)
    assert_in_delta(zero_zero, 0.0, 0.1)
    assert_in_delta(one_zero, 1.0, 0.1)
    assert_in_delta(one_one, 0.0, 0.1)
  end
end
