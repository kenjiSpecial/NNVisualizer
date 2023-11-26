// 数字をスライドすることができる
import { useEffect, useMemo, useState, useRef } from 'react';
import './App.css';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Line } from '@react-three/drei';
import mnist from 'mnist';
import { BufferGeometry, DoubleSide, MathUtils, Texture, Vector3 } from 'three';
import { button, useControls } from 'leva';
import Matrix from 'ml-matrix';
import { calcSigmoid, calcSoftMax, sigmoid } from './components/functions';
import { CameraMovement } from './components/camera-movement';

function App() {
  const inputSize = 28 * 28;
  const inputRowSize = 28;
  const inputColsize = 28;
  const hiddenSize = 50;
  const hiddenRowSize = 10;
  const hiddenColSize = 5;
  const outputSize = 10;
  const outputRowSize = 10;
  const outputColSize = 1;

  const inputPlane = { size: 1 / 14, space: 0.02 }; // const size = 1 / 14       const space = 0.02;
  const hiddenPlane = { size: 0.2, space: 0.08 };
  const outputPlane = { size: 1.2, space: 0.1 };

  const inputPos = new Vector3(0, 0, 0);
  const hiddenPos = new Vector3(0, 0, -2);
  const outputPos = new Vector3(0, 0, -4);

  const [data, setData] = useState<
    { W1: number[][]; b1: number[]; W2: number[][]; b2: number[] }[] | null
  >(null);
  // const count = 0;
  const testDataCnt = 100;
  const { count } = useControls({
    count: { value: 0, min: 0, max: testDataCnt - 1, step: 1 },
  });
  const testData = useMemo(() => mnist.set(0, testDataCnt).test, []);
  const paramData = useMemo(() => {
    if (!data) return null;
    return data[data?.length - 1];
  }, [data]);

  // インプットとparamsのW1をかけて、バイアスを足す
  const hiddenValue = useMemo(() => {
    if (!paramData || !testData) return null;

    const inp = new Matrix([testData[count].input]);
    const wMatrix = new Matrix(paramData.W1);
    const bMatrix = new Matrix([paramData.b1]);
    const res = inp.mmul(wMatrix).add(bMatrix);
    const sigmoidMatrix = calcSigmoid(res);
    return { wMatrix: wMatrix, hiddenValueMatrix: sigmoidMatrix };
  }, [paramData, testData, count]);
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
    fetch('/neural.json')
      .then((res) => res.json())
      .then((data) => {
        setData(data);
      });
  }, []);

  return (
    <>
      <div className="h-screen w-screen bg-slate-500 block" id="canvas">
        <Canvas camera={{ position: [2, 2, 8] }}>
          <ambientLight />
          <pointLight position={[5, 5, 5]} />

          <group position={[0, 0, 2]}>
            {testData ? (
              <PixelPlaneMesh
                renderOrder={4}
                position={inputPos}
                data={testData[count]}
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

            {paramArr ? (
              <DrawLineGroup
                inputSize={inputSize}
                hiddenSize={hiddenSize}
                outputSize={outputSize}
                paramW1Arr={paramArr.paramW1Arr}
                paramW2Arr={paramArr.paramW2Arr}
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

          <CameraMovement />
        </Canvas>
      </div>
    </>
  );
}

type PixelPlaneProps = JSX.IntrinsicElements['group'] & {
  data: { input: number[]; output: number[] };
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
  const data = props.data.input;
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
  halfRowSize: number;
  halfColSize: number;
  size: number;
  space: number;
}) {
  const { index, rowSize, halfRowSize, halfColSize, size, space } = props;
  const posX = ((index % rowSize) - halfRowSize) * (size + space);
  const posY = (-Math.floor(index / rowSize) + halfColSize) * (size + space);
  const posZ = 0;
  return { posX, posY, posZ };
}

function ParamsPixelPlaneMesh(props: ParamsPixelPlaneProps) {
  const arr = [];

  const { size, space } = props;
  const rowSize = props.rowSize;
  const halfRowSize = rowSize / 2;
  const halfColSize = props.colSize / 2;

  for (let i = 0; i < props.hiddenSize; i++) {
    const colorVal = Math.floor(props.hiddenValueArr[i] * 255);
    const color = `rgb(${colorVal}, ${colorVal}, ${colorVal})`;

    const { posX, posY, posZ } = hiddenPos({
      index: i,
      rowSize,
      halfRowSize,
      halfColSize,
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
  halfRowSize: number;
  size: number;
  space: number;
}) {
  const { index, rowSize, halfRowSize, size, space } = props;
  const posX = ((index % rowSize) - halfRowSize) * (size + space);
  const posY = 0;
  const posZ = 0;
  return { posX, posY, posZ };
}

function OutputMeshGroup(props: OutputMeshProps) {
  const outputMeshArr = [];
  const { size, space, rowsize } = props;
  const halfRowSize = rowsize / 2;
  for (let ii = 0; ii < props.outputSize; ii++) {
    const colorVal = Math.floor(props.outputValueArr[ii] * 255);
    const color = `rgb(${colorVal}, ${colorVal}, ${colorVal})`;
    const { posX, posY, posZ } = outputPos({
      index: ii,
      rowSize: rowsize,
      halfRowSize,
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
  paramW1Arr: number[][];
  paramW2Arr: number[][];
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
    paramW1Arr,
    paramW2Arr,
  } = props;
  // const [lineArr, setLineArr] = useState<JSX.Element[]>([]);
  const geometry = useRef<BufferGeometry>(null!);

  const { pos, color } = useMemo(() => {
    const arr = [];
    const colors = [];
    for (let i = 0; i < inputSize; i++) {
      const inputObj = inputPos({
        index: i,
        rowSize: input.rowSize,
        halfRowSize: input.rowSize / 2,
        size: input.size,
        space: input.space,
      });
      const inputPosVector = new Vector3(
        inputObj.posX,
        inputObj.posY,
        inputObj.posZ,
      );

      for (let ii = 0; ii < hiddenSize; ii++) {
        const hiddenObj = hiddenPos({
          index: ii,
          rowSize: hidden.rowSize,
          halfRowSize: hidden.rowSize / 2,
          halfColSize: hidden.colSize / 2,
          size: hidden.size,
          space: hidden.space,
        });

        const hiddenPosVector = new Vector3(
          hiddenObj.posX,
          hiddenObj.posY,
          hiddenObj.posZ - 2,
        );

        const color = paramW1Arr[i][ii] > 0.6 ? paramW1Arr[i][ii] * 2 : 0.1;
        const alpha =
          paramW1Arr[i][ii] > 0.6 ? paramW1Arr[i][ii] * 0.05 : 0.005;

        arr.push(inputPosVector.x);
        arr.push(inputPosVector.y);
        arr.push(inputPosVector.z);
        arr.push(hiddenPosVector.x);
        arr.push(hiddenPosVector.y);
        arr.push(hiddenPosVector.z);

        colors.push(color);
        colors.push(color);
        colors.push(color);
        colors.push(alpha);
        colors.push(color);
        colors.push(color);
        colors.push(color);
        colors.push(alpha);
      }
    }

    for (let i = 0; i < hiddenSize; i++) {
      const hiddenObj = hiddenPos({
        index: i,
        rowSize: hidden.rowSize,
        halfRowSize: hidden.rowSize / 2,
        halfColSize: hidden.colSize / 2,
        size: hidden.size,
        space: hidden.space,
      });
      const hiddenPosVector = new Vector3(
        hiddenObj.posX,
        hiddenObj.posY,
        hiddenObj.posZ - 2,
      );
      for (let j = 0; j < outputSize; j++) {
        const { posX, posY, posZ } = outputPos({
          index: j,
          rowSize: output.rowSize,
          halfRowSize: output.rowSize / 2,
          size: output.size,
          space: output.space,
        });
        const outputPosVector = new Vector3(posX, posY, posZ - 4);

        const color = Math.pow(paramW2Arr[i][j], 2);
        const alpha = Math.pow(paramW2Arr[i][j], 1) * 0.5;

        arr.push(hiddenPosVector.x);
        arr.push(hiddenPosVector.y);
        arr.push(hiddenPosVector.z);
        arr.push(outputPosVector.x);
        arr.push(outputPosVector.y);
        arr.push(outputPosVector.z);

        colors.push(color);
        colors.push(color);
        colors.push(color);
        colors.push(alpha);
        colors.push(color);
        colors.push(color);
        colors.push(color);
        colors.push(alpha);
      }
    }

    if (geometry.current) {
      geometry.current.attributes.position.needsUpdate = true;
      geometry.current.attributes.color.needsUpdate = true;
    }
    return { pos: new Float32Array(arr), color: new Float32Array(colors) };
  }, []);

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
