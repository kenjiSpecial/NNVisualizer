import { Matrix } from 'ml-matrix';

/**
 * アフィン変換の計算を行う
 *
 * @param  input
 * @param  weight
 * @param  bias
 * @returns
 */
export function calcAffine(input: Matrix, weight: Matrix, bias: Matrix) {
  if (bias.rows !== 1) {
    throw new Error('バイアスの行数が1ではありません');
  }
  if (input.columns !== weight.rows) {
    throw new Error('入力の列数と重みの行数が一致しません');
  }

  return input.clone().mmul(weight).addRowVector(bias);
}

/**
 * シグモイド関数を計算する
 *
 * @param {Matrix} input
 *
 * @returns {Matrix}
 */
export function calcSigmoid(input: Matrix) {
  // return nj.sigmoid(input);
  const output = input.clone();
  for (let i = 0; i < input.rows; i++) {
    for (let j = 0; j < input.columns; j++) {
      output.set(i, j, 1 / (1 + Math.exp(-output.get(i, j))));
    }
  }

  return output;
}

/**
 *
 * @param input
 *
 * @returns
 */
export function calcSoftMax(input: Matrix) {
  const inputArr = input.to2DArray();
  const outputArr = [];
  for (let i = 0; i < inputArr.length; i++) {
    const valArr = inputArr[i];
    const xMax = Math.max(...valArr);
    let sum = 0;
    let arr = [];
    for (let j = 0; j < valArr.length; j++) {
      // console.log(valArr);
      const val = valArr[j];
      const exVal = Math.exp(val - xMax);
      // console.log(exVal);
      sum += exVal;
      arr.push(exVal);
    }
    arr = arr.map((val) => val / sum);
    outputArr.push(arr);
  }

  return new Matrix(outputArr);
}

/**
 *
 * 交差エントロピー誤差を計算する
 *
 * @param  inputArr
 * @param  tArr
 *
 * @returns
 */
export function calcCrossEntrypyError(inputArr: Matrix, tArr: Matrix) {
  const batchSize = inputArr.rows;
  const softmaxMatrix = calcSoftMax(inputArr);
  return -Matrix.mul(tArr, Matrix.log(softmaxMatrix)).sum() / batchSize;
}

/**
 *
 * 勾配を求める
 *
 * @param loss () => number
 * @param weight Matrix
 * @returns Matrix
 */
export function numericalGradient(loss: () => number, weight: Matrix) {
  const h = 1e-4;
  const grad = new Matrix(weight.rows, weight.columns);
  for (let i = 0; i < weight.rows; i++) {
    //weight.rows; i++) {
    for (let j = 0; j < weight.columns; j++) {
      console.log(`${i}, ${j}`);
      const tmpVal = weight.get(i, j);
      weight.set(i, j, tmpVal + h);
      const fxh1 = loss();
      weight.set(i, j, tmpVal - h);
      const fxh2 = loss();
      grad.set(i, j, (fxh1 - fxh2) / (2 * h));
      weight.set(i, j, tmpVal);
    }
  }

  return grad;
}

export function sigmoid(x: number) {
  return 1 / (1 + Math.exp(-x));
}
