import { Matrix } from 'ml-matrix';
import { calcAffine, calcSigmoid, calcSoftMax } from './functions';

// 試しにテストを書いてみる
test('chcek', () => {
  expect(1).toBe(1);
});

// アフィン変換の計算のテストを書いてみる
test('affine transformation', () => {
  const input = new Matrix([
    [1, 0],
    [0, 1],
    [1, 1],
  ]);
  const weight = new Matrix([
    [1, 1, 0],
    [0, 1, 1],
  ]);
  const bias = new Matrix([[1, 1, 1]]);
  const result = calcAffine(input, weight, bias);
  const expected = new Matrix([
    [2, 2, 1],
    [1, 2, 2],
    [2, 3, 2],
  ]);
  expect(result).toEqual(expected);
});

// シグモイド関数のテストを書いてみる
test('sigmoid function', () => {
  const input = new Matrix([[1, 2, 3]]);
  const result = calcSigmoid(input);
  const expected = new Matrix([
    [0.7310585786300049, 0.8807970779778823, 0.9525741268224334],
  ]);
  expect(result).toEqual(expected);
});

// ソフトマックス関数のテストを書いてみる
test('softmax function', () => {
  const input = new Matrix([
    [1, 2, 3],
    [4, 5, 6],
  ]);
  const result = calcSoftMax(input);
  const expected = new Matrix([
    [0.09003057317038046, 0.24472847105479764, 0.6652409557748218],
    [0.09003057317038046, 0.24472847105479764, 0.6652409557748218],
  ]);
  expect(result).toEqual(expected);
});
