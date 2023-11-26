/// <reference types="vite/client" />

declare module 'mnist' {
  type Output = [
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
  ];

  type Datum = {
    input: number[];
    output: Output;
  };

  interface Digit {
    id: number;
    raw: number[];
    length: number;
    get: (index?: number) => number[];
    range: (start: number, end: number) => number[][];
    set: (start: number, end: number) => Datum[];
  }

  namespace MNIST {
    export function set(
      trainingSetSize: number,
      testSetSize: number,
    ): {
      training: Datum[];
      test: Datum[];
    };

    export function get(count: number): Datum[];

    export function draw(
      digit: number[],
      context: CanvasRenderingContext2D,
      offsetX: number,
      offsetY: number,
    ): void;

    export function toNumber(array: number[]): number;
  }

  export = MNIST as typeof MNIST & Digit[];
}

// declare module 'numjs' {
//   export interface NdArray<T = any> {
//     log: () => NdArray;
//   }
// }
