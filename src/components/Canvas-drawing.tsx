import React, { useRef, useEffect, useMemo } from 'react';

export const CanvasDrawing: React.FC<{
  setInputData: (data: number[]) => void;
  inputData: number[];
  isButtonClicked: React.MutableRefObject<boolean>;
  windowWidth: number | undefined;
  parentRef: React.RefObject<HTMLDivElement>;
}> = ({ setInputData, inputData, isButtonClicked, windowWidth, parentRef }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // 28 * 28の2D Contextを作成する
  const vContext = useMemo(() => {
    const vCanvas = document.createElement('canvas');
    vCanvas.width = 28;
    vCanvas.height = 28;
    return vCanvas.getContext('2d', { willReadFrequently: true });
  }, []);
  const pt = useRef({ x: 0, y: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context) return;
    if (!vContext) return;
    const data = vContext.getImageData(0, 0, 28, 28);
    for (let i = 0; i < data.data.length; i++) {
      const val = Math.floor(255 * inputData[i]);
      data.data[4 * i] = val; //Math.floor(inputData[i] * 255);
      data.data[4 * i + 1] = val; //inputData[i] * 255;
      data.data[4 * i + 2] = val; //inputData[i] * 255;
      data.data[4 * i + 3] = 255; //inputData[i] * 255;
    }
    vContext.putImageData(data, 0, 0);
    context.drawImage(vContext.canvas, 0, 0, canvas.width, canvas.height);
  }, []);

  useEffect(() => {
    if (!isButtonClicked.current) return;

    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context) return;
    if (!vContext) return;
    const data = vContext.getImageData(0, 0, 28, 28);
    for (let i = 0; i < data.data.length; i++) {
      const val = Math.floor(255 * inputData[i]);
      data.data[4 * i] = val; //Math.floor(inputData[i] * 255);
      data.data[4 * i + 1] = val; //inputData[i] * 255;
      data.data[4 * i + 2] = val; //inputData[i] * 255;
      data.data[4 * i + 3] = 255; //inputData[i] * 255;
    }
    vContext.putImageData(data, 0, 0);
    context.drawImage(vContext.canvas, 0, 0, canvas.width, canvas.height);

    isButtonClicked.current = false;
  }, [isButtonClicked.current, inputData, canvasRef.current, vContext]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const context = canvas.getContext('2d');
    if (!context) return;
    // canvasをblackで塗りつぶす

    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    const startDrawing = (event: MouseEvent | TouchEvent) => {
      isDrawing = true;
      const { clientX, clientY } =
        event instanceof MouseEvent ? event : event.touches[0];
      const rect = canvas.getBoundingClientRect();
      lastX = clientX - rect.left;
      lastY = clientY - rect.top;

      pt.current = { x: lastX, y: lastY };

      context.fillStyle = 'black';
      context.fillRect(0, 0, canvas.width, canvas.height);

      if (event instanceof TouchEvent) {
        event.preventDefault();
      }
    };

    const draw = (event: MouseEvent | TouchEvent) => {
      if (!isDrawing) return;
      const { clientX, clientY } =
        event instanceof MouseEvent ? event : event.touches[0];
      const rect = canvas.getBoundingClientRect();
      const x = clientX - rect.left;
      const y = clientY - rect.top;
      pt.current = { x, y };

      // 滑らかに線を描画する
      context.lineJoin = 'round';
      context.lineCap = 'round';

      context.beginPath();
      context.moveTo(lastX, lastY);
      context.lineTo(x, y);
      context.strokeStyle = 'white';
      context.lineWidth = event instanceof MouseEvent ? 40 : 20;
      context.stroke();
      context.closePath();

      lastX = x;
      lastY = y;

      // 28 * 28にリサイズする
      vContext!.drawImage(canvas, 0, 0, 28, 28);
      // 28 * 28のImageDataを取得する
      const imageData = vContext!.getImageData(0, 0, 28, 28);
      // 28 * 28のImageDataを白黒データを正規規化して、入力データにする
      // 配列のおおきさは 28 * 28  = 784このデータが入力データになる
      const normalizedData = Array.from(imageData.data)
        .filter((_, index) => index % 4 === 0)
        .map((v) => v / 255);

      setInputData(normalizedData);

      if (event instanceof TouchEvent) {
        event.preventDefault();
      }
    };

    const stopDrawing = () => {
      isDrawing = false;
    };

    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('touchstart', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('touchmove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('touchend', stopDrawing);
    // resize

    return () => {
      canvas.removeEventListener('mousedown', startDrawing);
      canvas.removeEventListener('touchstart', startDrawing);
      canvas.removeEventListener('mousemove', draw);
      canvas.removeEventListener('touchmove', draw);
      canvas.removeEventListener('mouseup', stopDrawing);
      canvas.removeEventListener('touchend', stopDrawing);
    };
  }, []);

  const canvasWidth = useMemo(() => {
    // console.log(parentRef.current?.offsetWidth);
    if (!windowWidth) {
      return window.innerWidth < 640 ? 200 : 320;
    }
    if (windowWidth < 640) {
      return 200;
    } else {
      return 320;
    }
  }, [windowWidth, parentRef]);

  return (
    <>
      <canvas
        ref={canvasRef}
        width={canvasWidth}
        height={canvasWidth}
        className="sm:w-full cursor-pointer"
      />
    </>
  );
};
