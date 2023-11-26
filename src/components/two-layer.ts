import { Matrix } from 'ml-matrix';
import {
  calcAffine,
  calcSigmoid,
  calcCrossEntrypyError,
  calcSoftMax,
  numericalGradient,
} from './functions';

export class TwoLayerNet {
  params: {
    W1: Matrix;
    b1: Matrix;
    W2: Matrix;
    b2: Matrix;
  };
  constructor(
    inputSize: number,
    hiddenSize: number,
    outputSize: number,
    weightInitStd = 0.01,
  ) {
    this.params = {
      W1: Matrix.rand(inputSize, hiddenSize).mul(weightInitStd),
      b1: Matrix.zeros(1, hiddenSize),
      W2: Matrix.rand(hiddenSize, outputSize).mul(weightInitStd),
      b2: Matrix.zeros(1, outputSize),
    };
    console.log(this.params.W1.to2DArray());
    console.log(this.params.b1.to2DArray());
    // console.log(this.params.W1.tolist());
  }

  /**
   *  予測を行う
   *
   * @param x データ (N(データ数) x D(入力層のノード数))
   */
  predict(x: Matrix) {
    const { W1, W2, b1, b2 } = this.params;

    const z1 = calcSigmoid(calcAffine(x, W1, b1));
    const a2 = calcAffine(z1, W2, b2);
    const y = calcSoftMax(a2);

    return y;
  }

  loss(x: Matrix, t: Matrix) {
    const y = this.predict(x);
    const loss = calcCrossEntrypyError(y, t);

    return loss;
  }

  numericalGradient(xBatch: Matrix, tBatch: Matrix) {
    const lossW = () => {
      return this.loss(xBatch, tBatch);
    };

    const grads = {
      W1: numericalGradient(lossW, this.params.W1),
      // b1: numericalGradient(lossW, this.params.b1),
      // W2: numericalGradient(lossW, this.params.W2),
      // b2: numericalGradient(lossW, this.params.b2),
    };
    // console.log(grads.W1);

    return grads;
  }
}
