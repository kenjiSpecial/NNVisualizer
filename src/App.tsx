// 数字をスライドすることができる
import { useEffect, useMemo, useState, useRef, useCallback } from 'react';
import './App.css';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { BufferGeometry, DoubleSide, MathUtils, Texture, Vector3 } from 'three';
import Matrix from 'ml-matrix';
import { calcSigmoid, calcSoftMax } from './components/functions';
import { CanvasDrawing } from './components/Canvas-drawing';
import { ministSampleData } from './components/mnist-sample';
import { useWindowSize } from './components/use-window-size';

function App() {
  const autoNumCntRef = useRef(
    Math.floor(Math.random() * ministSampleData.length),
  );
  const isButtonClicked = useRef(false);

  const [inputData, setInputData] = useState<number[]>(
    ministSampleData[autoNumCntRef.current],
  );
  const ref = useRef<HTMLDivElement>(null);
  const windowSize = useWindowSize();

  const inputSize = 28 * 28;
  const inputRowSize = 28;
  const inputColsize = 28;
  const hiddenSize = 50;
  const hiddenRowSize = 10;
  const hiddenColSize = 5;
  const outputSize = 10;
  const outputRowSize = 3;
  const outputColSize = 4;

  const inputPlane = { size: 1 / 10, space: 0.02 }; // const size = 1 / 14       const space = 0.02;
  const hiddenPlane = { size: 0.3, space: 0.1 };
  const outputPlane = { size: 2, space: 0.3 };

  const inputPos = new Vector3(0, 0, 0);
  const hiddenPos = new Vector3(0, 0, -4);
  const outputPos = new Vector3(0, 0, -8);

  const [data, setData] = useState<{
    W1: number[][];
    b1: number[];
    W2: number[][];
    b2: number[];
  } | null>(null);

  const isLoading = useMemo(() => {
    return data === null;
  }, [data]);
  const displayLoading = useMemo(() => {
    return isLoading ? 'block' : 'hidden';
  }, [isLoading]);
  const displayCanvas = useMemo(() => {
    return isLoading ? 'hidden' : 'block';
  }, [isLoading]);
  const displaySideBar = useMemo(() => {
    return isLoading ? 'hidden' : 'flex';
  }, [isLoading]);

  const paramData = useMemo(() => {
    if (!data) return null;
    return data;
  }, [data]);

  // インプットとparamsのW1をかけて、バイアスを足す
  const hiddenValue = useMemo(() => {
    if (!paramData || !inputData) return null;

    const inp = new Matrix([inputData]);
    const wMatrix = new Matrix(paramData.W1);
    const bMatrix = new Matrix([paramData.b1]);
    const res = inp.mmul(wMatrix).add(bMatrix);
    const sigmoidMatrix = calcSigmoid(res);
    return { wMatrix: wMatrix, hiddenValueMatrix: sigmoidMatrix };
  }, [paramData, inputData]);
  const hiddenValueMatrix = hiddenValue?.hiddenValueMatrix;
  const inputHiddenWMatrix = hiddenValue?.wMatrix;

  const outputValue = useMemo(() => {
    if (!paramData || !hiddenValueMatrix) return null;

    const wMatrix = new Matrix(paramData.W2);
    const bMatrix = new Matrix([paramData.b2]);
    const res = hiddenValueMatrix.mmul(wMatrix).add(bMatrix);
    const softMaxMatrix = calcSoftMax(res);
    return { wMatrix: wMatrix, outputValueMatrix: softMaxMatrix };
  }, [paramData, hiddenValueMatrix]);
  const outputValueMatrix = outputValue?.outputValueMatrix;
  const hiddenOutputWMatrix = outputValue?.wMatrix;

  const paramArr = useMemo(() => {
    if (!hiddenOutputWMatrix || !inputHiddenWMatrix) return null;
    const inputHiddennArr = inputHiddenWMatrix.to1DArray();
    const minInputHiddenVal = Math.min(...inputHiddennArr);
    const maxInputHiddenVal = Math.max(...inputHiddennArr);
    const paramW1Arr = [];
    for (let i = 0; i < inputHiddenWMatrix.rows; i++) {
      const arr = [];
      for (let ii = 0; ii < inputHiddenWMatrix.columns; ii++) {
        arr.push(
          MathUtils.clamp(
            (inputHiddenWMatrix.get(i, ii) - minInputHiddenVal) /
              (maxInputHiddenVal - minInputHiddenVal),
            0,
            1,
          ),
        );
      }
      paramW1Arr.push(arr);
    }

    const hiddenOutputArr = hiddenOutputWMatrix.to1DArray();
    const minHiddenOutputVal = Math.min(...hiddenOutputArr);
    const maxHiddenOutputVal = Math.max(...hiddenOutputArr);
    const paramW2Arr = [];
    for (let i = 0; i < hiddenOutputWMatrix.rows; i++) {
      const arr = [];
      for (let ii = 0; ii < hiddenOutputWMatrix.columns; ii++) {
        arr.push(
          MathUtils.clamp(
            (hiddenOutputWMatrix.get(i, ii) - minHiddenOutputVal) /
              (maxHiddenOutputVal - minHiddenOutputVal),
            0,
            1,
          ),
        );
      }
      paramW2Arr.push(arr);
    }

    return { paramW1Arr, paramW2Arr };
  }, [inputHiddenWMatrix, hiddenOutputWMatrix]);

  useEffect(() => {
    fetch('/neural-param.json')
      .then((res) => res.json())
      .then((data) => {
        setData(data);
      });
  }, []);

  const clickbutton = useCallback(() => {
    autoNumCntRef.current = Math.floor(Math.random() * ministSampleData.length);
    isButtonClicked.current = true;
    setInputData(ministSampleData[autoNumCntRef.current]);
  }, []);

  return (
    <>
      <div
        className="h-screen  w-screen bg-gradient-to-t from-orange-100 to-blue-400 block cursor-grab "
        id="canvas"
      >
        <div className={`${displayCanvas} w-full h-2/3 sm:h-screen`}>
          <Canvas camera={{ position: [2, 2, 8] }}>
            <ambientLight />
            <pointLight position={[5, 5, 5]} />

            <group position={[0, 0, 2]}>
              {inputData ? (
                <PixelPlaneMesh
                  renderOrder={4}
                  position={inputPos}
                  data={inputData}
                  size={inputPlane.size}
                  space={inputPlane.space}
                  rowSize={inputRowSize}
                  colSize={inputColsize}
                />
              ) : null}

              {hiddenValueMatrix ? (
                <ParamsPixelPlaneMesh
                  renderOrder={3}
                  position={hiddenPos}
                  hiddenSize={hiddenSize}
                  hiddenValueArr={hiddenValueMatrix.to1DArray()}
                  size={hiddenPlane.size}
                  space={hiddenPlane.space}
                  rowSize={hiddenRowSize}
                  colSize={hiddenColSize}
                />
              ) : null}

              {outputValueMatrix ? (
                <OutputMeshGroup
                  renderOrder={2}
                  position={outputPos}
                  outputSize={outputSize}
                  outputValueArr={outputValueMatrix.to1DArray()}
                  size={outputPlane.size}
                  space={outputPlane.space}
                  rowsize={outputRowSize}
                  colSize={outputColSize}
                />
              ) : null}

              {inputData &&
              paramArr &&
              hiddenValueMatrix &&
              outputValueMatrix ? (
                <DrawLineGroup
                  inputSize={inputSize}
                  hiddenSize={hiddenSize}
                  outputSize={outputSize}
                  inputPos={inputPos}
                  hiddenPos={hiddenPos}
                  outputPos={outputPos}
                  paramW1Arr={paramArr.paramW1Arr}
                  paramW2Arr={paramArr.paramW2Arr}
                  input1DArr={inputData}
                  hidden1DArr={hiddenValueMatrix.to1DArray()}
                  output1DArr={outputValueMatrix.to1DArray()}
                  renderOrder={1}
                  input={
                    {
                      size: inputPlane.size,
                      space: inputPlane.space,
                      rowSize: inputRowSize,
                      colSize: inputColsize,
                    } as drawLineGroup['input']
                  }
                  hidden={{
                    size: hiddenPlane.size,
                    space: hiddenPlane.space,
                    rowSize: hiddenRowSize,
                    colSize: hiddenColSize,
                  }}
                  output={{
                    size: outputPlane.size,
                    space: outputPlane.space,
                    rowSize: outputRowSize,
                    colSize: outputColSize,
                  }}
                />
              ) : null}
            </group>

            <OrbitControls />
          </Canvas>
        </div>
        {/* 画面の中央にLOADINGを表示する */}
        <div
          className={`${displayLoading} absolute top-0 left-0 w-full h-full`}
        >
          <div className="flex items-center justify-center h-screen">
            <div className="text-4xl font-bold text-white">LOADING</div>
          </div>
        </div>
      </div>

      <div
        className="
          fixed 
          sm:flex sm:flex-col sm:justify-between 
          left-0 
          bottom-0 sm:top-0 
          sm:h-full 
          w-full sm:w-96 
          bg-gray-100 z-10 bg-opacity-50 
          p-3
          sm:p-8 border-r border-gray-400
      "
      >
        <div className="hidden sm:block">
          <div className="text-2xl font-bold mb-2 ">
            <p>NNビジュアライザー</p>
          </div>

          <div className="text-base">
            <p>
              <a
                href="https://amzn.asia/d/fKFUgBV"
                target="_blank"
                className="text-blue-700 underline hover:text-blue-900"
              >
                ゼロから作るDeep Learning
              </a>
              の第4章のニューラルネットワークをリアルタイムに可視化しています。
            </p>
            <p>
              入力層756個、隠れ層50個、出力層10個のニューラルネットワークを可視化しています。
            </p>
          </div>
        </div>

        <div className={`${displaySideBar} flex-col justify-end`} ref={ref}>
          <div className="text-base sm:text-lg font-bold mb-1 sm:mb-2">
            スケッチキャンバス
          </div>
          <div className="mb-2 sm:mb-4">
            <div className="mb-2">
              <CanvasDrawing
                setInputData={setInputData}
                inputData={inputData}
                isButtonClicked={isButtonClicked}
                windowWidth={windowSize.width}
                parentRef={ref}
              />
            </div>
            <div className="text-left sm:text-right">
              <button
                className="bg-blue-500 hover:bg-blue-700 text-sm sm:text-base text-white font-bold py-2 px-4 rounded"
                onClick={clickbutton}
              >
                画像テスト
              </button>
            </div>
          </div>
          <div className="text-xs sm:text-base mb-2 sm:mb-4 block">
            <p>
              スケッチキャンバスにスケッチをするか、画像テストボタンを押してください。自動でニューラルネットワークが推論を行います。
            </p>
            <p>スケッチは一筆書きになっています。</p>
          </div>
          <div>
            <p className="text-sm sm:text-base text-right">
              Code:{' '}
              <a
                href="https://github.com/kenjiSpecial/NNVisualizer"
                className="text-blue-700 underline hover:text-blue-900"
                target="_blank"
              >
                kenjiSpecial/NNVisualizer
              </a>
            </p>
            <p></p>
          </div>
        </div>
      </div>
    </>
  );
}

type PixelPlaneProps = JSX.IntrinsicElements['group'] & {
  data: number[];
  size: number;
  space: number;
  rowSize: number;
  colSize: number;
};

function inputPos(props: {
  index: number;
  rowSize: number;
  halfRowSize: number;
  size: number;
  space: number;
}) {
  const { index, rowSize, halfRowSize, size, space } = props;
  const posX = ((index % rowSize) - halfRowSize) * (size + space);
  const posY = (-Math.floor(index / rowSize) + halfRowSize) * (size + space);
  const posZ = 0;
  return { posX, posY, posZ };
}

function PixelPlaneMesh(props: PixelPlaneProps) {
  // const data = [1, 2, 3, 4];
  const data = props.data;
  const { size, space, rowSize } = props;
  const halfRowSize = rowSize / 2;

  return (
    <group {...props}>
      {data.map((d, index) => {
        const colorVal = Math.floor(d * 255);
        const color = `rgb(${colorVal}, ${colorVal}, ${colorVal})`;

        const { posX, posY, posZ } = inputPos({
          index,
          rowSize,
          halfRowSize,
          size,
          space,
        });
        const key = `plane-${index}`;

        return (
          <mesh
            key={key}
            scale={[size, size, size]}
            position={[posX, posY, posZ]}
          >
            <planeGeometry args={[1, 1]} />
            <meshBasicMaterial color={color} side={DoubleSide} />
          </mesh>
        );
      })}
    </group>
  );
}

type ParamsPixelPlaneProps = JSX.IntrinsicElements['group'] & {
  hiddenSize: number;
  hiddenValueArr: number[];
  size: number;
  space: number;
  rowSize: number;
  colSize: number;
};

function hiddenPos(props: {
  index: number;
  rowSize: number;
  colSize: number;
  size: number;
  space: number;
}) {
  const { index, rowSize, colSize, size, space } = props;
  const posX = ((index % rowSize) - (rowSize - 1) / 2) * (size + space);
  const posY =
    (-Math.floor(index / rowSize) + (colSize - 1) / 2) * (size + space);
  const posZ = 0;
  return { posX, posY, posZ };
}

function ParamsPixelPlaneMesh(props: ParamsPixelPlaneProps) {
  const arr = [];

  const { size, space } = props;
  const rowSize = props.rowSize;
  const colSize = props.colSize;

  for (let i = 0; i < props.hiddenSize; i++) {
    const colorVal = Math.floor(props.hiddenValueArr[i] * 255);
    const color = `rgb(${colorVal}, ${colorVal}, ${colorVal})`;

    const { posX, posY, posZ } = hiddenPos({
      index: i,
      rowSize,
      colSize,
      size,
      space,
    });
    const key = `plane-${i}`;
    arr.push(
      <mesh key={key} scale={[size, size, size]} position={[posX, posY, posZ]}>
        <planeGeometry args={[1, 1]} />
        <meshBasicMaterial color={color} side={DoubleSide} />
      </mesh>,
    );
  }
  return <group {...props}>{arr}</group>;
}

type OutputMeshProps = JSX.IntrinsicElements['group'] & {
  outputSize: number;
  outputValueArr: number[];
  size: number;
  space: number;
  rowsize: number;
  colSize: number;
};

function outputPos(props: {
  index: number;
  rowSize: number;
  colSize: number;
  size: number;
  space: number;
}) {
  const { index, rowSize, size, space, colSize } = props;
  const newIndex = index === 0 ? 9 : index - 1;
  const posX = ((newIndex % rowSize) - (rowSize - 1) / 2) * (size + space);
  const posY =
    (-Math.floor(newIndex / rowSize) + (colSize - 1) / 2) * (size + space);
  const posZ = 0;
  return { posX, posY, posZ };
}

function OutputMeshGroup(props: OutputMeshProps) {
  const outputMeshArr = [];
  const { size, space, rowsize } = props;
  for (let ii = 0; ii < props.outputSize; ii++) {
    const colorVal = Math.floor(props.outputValueArr[ii] * 255);
    const color = `rgb(${colorVal}, ${colorVal}, ${colorVal})`;
    const { posX, posY, posZ } = outputPos({
      index: ii,
      rowSize: rowsize,
      colSize: props.colSize,
      size,
      space,
    });
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.fillStyle = color;
      ctx.fillRect(0, 0, 64, 64);
      ctx.fillStyle = 'black';
      ctx.font = '40px san-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`${ii}`, 32, 32);
    }
    const texture = new Texture(canvas);
    texture.needsUpdate = true;
    const key = `plane-${ii}`;
    outputMeshArr.push(
      <mesh key={key} scale={[size, size, size]} position={[posX, posY, posZ]}>
        <planeGeometry args={[1, 1]} />
        <meshBasicMaterial map={texture} side={DoubleSide} />
      </mesh>,
    );
  }

  return <group {...props}>{outputMeshArr}</group>;
}

type drawLineGroup = JSX.IntrinsicElements['group'] & {
  inputSize: number;
  hiddenSize: number;
  outputSize: number;
  inputPos: Vector3;
  hiddenPos: Vector3;
  outputPos: Vector3;
  paramW1Arr: number[][];
  paramW2Arr: number[][];
  input1DArr: number[];
  hidden1DArr: number[];
  output1DArr: number[];
  input: {
    size: number;
    space: number;
    rowSize: number;
    colSize: number;
  };
  hidden: {
    size: number;
    space: number;
    rowSize: number;
    colSize: number;
  };
  output: {
    size: number;
    space: number;
    rowSize: number;
    colSize: number;
  };
};

function DrawLineGroup(props: drawLineGroup) {
  const {
    input,
    hidden,
    output,
    inputSize,
    hiddenSize,
    outputSize,
    input1DArr,
    hidden1DArr,
    output1DArr,
  } = props;
  // const [lineArr, setLineArr] = useState<JSX.Element[]>([]);
  const geometry = useRef<BufferGeometry>(null!);

  const pos = useMemo(() => {
    const arr = [];
    for (let i = 0; i < inputSize; i++) {
      const inputObj = inputPos({
        index: i,
        rowSize: input.rowSize,
        halfRowSize: input.rowSize / 2,
        size: input.size,
        space: input.space,
      });
      const inputPosVector = new Vector3(
        inputObj.posX + props.inputPos.x,
        inputObj.posY + props.inputPos.y,
        inputObj.posZ + props.inputPos.z,
      );

      for (let ii = 0; ii < hiddenSize; ii++) {
        const hiddenObj = hiddenPos({
          index: ii,
          rowSize: hidden.rowSize,
          colSize: hidden.colSize,
          size: hidden.size,
          space: hidden.space,
        });

        const hiddenPosVector = new Vector3(
          hiddenObj.posX + props.hiddenPos.x,
          hiddenObj.posY + props.hiddenPos.y,
          hiddenObj.posZ + props.hiddenPos.z,
        );

        const randInputRad = (input.size / 2) * MathUtils.randFloat(0.1, 0.9);
        const randHiddenRad = (hidden.size / 2) * MathUtils.randFloat(0.1, 0.9);
        const randInputAngle = MathUtils.randFloat(0, Math.PI * 2);
        const randHiddenAngle = MathUtils.randFloat(0, Math.PI * 2);
        const randInputX = randInputRad * Math.cos(randInputAngle);
        const randInputY = randInputRad * Math.sin(randInputAngle);
        const randHiddenX = randHiddenRad * Math.cos(randHiddenAngle);
        const randHiddenY = randHiddenRad * Math.sin(randHiddenAngle);

        arr.push(inputPosVector.x + randInputX);
        arr.push(inputPosVector.y + randInputY);
        arr.push(inputPosVector.z);
        arr.push(hiddenPosVector.x + randHiddenX);
        arr.push(hiddenPosVector.y + randHiddenY);
        arr.push(hiddenPosVector.z);
      }
    }

    for (let i = 0; i < hiddenSize; i++) {
      const hiddenObj = hiddenPos({
        index: i,
        rowSize: hidden.rowSize,
        colSize: hidden.colSize,
        size: hidden.size,
        space: hidden.space,
      });
      const hiddenPosVector = new Vector3(
        hiddenObj.posX + props.hiddenPos.x,
        hiddenObj.posY + props.hiddenPos.y,
        hiddenObj.posZ + props.hiddenPos.z,
      );
      for (let j = 0; j < outputSize; j++) {
        const { posX, posY, posZ } = outputPos({
          index: j,
          rowSize: output.rowSize,
          colSize: output.colSize,
          size: output.size,
          space: output.space,
        });
        const outputPosVector = new Vector3(
          posX + props.outputPos.x,
          posY + props.outputPos.y,
          posZ + props.outputPos.z,
        );
        const randHiddenRad = (hidden.size / 2) * MathUtils.randFloat(0.1, 0.9);
        const randOutputRad = (output.size / 2) * MathUtils.randFloat(0.1, 0.9);
        const randHiddenAngle = MathUtils.randFloat(0, Math.PI * 2);
        const randOutputAngle = MathUtils.randFloat(0, Math.PI * 2);
        const randHiddenX = randHiddenRad * Math.cos(randHiddenAngle);
        const randHiddenY = randHiddenRad * Math.sin(randHiddenAngle);
        const randOutputX = randOutputRad * Math.cos(randOutputAngle);
        const randOutputY = randOutputRad * Math.sin(randOutputAngle);

        arr.push(hiddenPosVector.x + randHiddenX);
        arr.push(hiddenPosVector.y + randHiddenY);
        arr.push(hiddenPosVector.z);
        arr.push(outputPosVector.x + randOutputX);
        arr.push(outputPosVector.y + randOutputY);
        arr.push(outputPosVector.z);
      }
    }

    if (geometry.current) {
      geometry.current.attributes.position.needsUpdate = true;
    }
    return new Float32Array(arr);
  }, []);

  const color = useMemo(() => {
    const colors = [];
    for (let i = 0; i < inputSize; i++) {
      const color1 = input1DArr[i];
      const alpha1 = MathUtils.clamp(input1DArr[i] * 0.3, 0.02, 1);

      for (let ii = 0; ii < hiddenSize; ii++) {
        const color2 = hidden1DArr[ii];
        const alpha2 = hidden1DArr[ii] * hidden1DArr[ii] * 0.1;

        colors.push(
          color1,
          color1,
          color1,
          alpha1,
          color2,
          color2,
          color2,
          alpha2,
        );
      }
    }

    for (let i = 0; i < hiddenSize; i++) {
      const color1 = hidden1DArr[i];
      const alpha1 = MathUtils.clamp(hidden1DArr[i], 0.1, 1);
      for (let j = 0; j < outputSize; j++) {
        const color2 = output1DArr[j];
        const alpha2 = MathUtils.clamp(output1DArr[j] * 3, 0.01, 1);

        colors.push(
          color1,
          color1,
          color1,
          alpha1,
          color2,
          color2,
          color2,
          alpha2,
        );
      }
    }

    if (geometry.current) {
      geometry.current.attributes.color.needsUpdate = true;
    }
    return new Float32Array(colors);
  }, [input1DArr, hidden1DArr, output1DArr]);

  return (
    <lineSegments>
      <bufferGeometry ref={geometry}>
        <bufferAttribute
          attach={'attributes-position'}
          count={pos.length / 3}
          itemSize={3}
          array={pos}
        />
        <bufferAttribute
          attach={'attributes-color'}
          count={color.length / 4}
          itemSize={4}
          array={color}
        />
      </bufferGeometry>
      <lineBasicMaterial
        attach="material"
        vertexColors={true}
        linewidth={0.1}
        transparent={true}
        depthTest={true}
        depthWrite={false}
      />
    </lineSegments>
  );
}

export default App;
