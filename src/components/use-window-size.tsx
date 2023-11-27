import { useState, useEffect } from 'react';

// ウィンドウサイズを表す型定義
interface WindowSize {
  width: number | undefined;
  height: number | undefined;
}

// カスタムフックの定義
export const useWindowSize = (): WindowSize => {
  const [windowSize, setWindowSize] = useState<WindowSize>({
    width: undefined,
    height: undefined,
  });

  useEffect(() => {
    // サイズを設定するヘルパー関数
    const handleResize = () => {
      setWindowSize({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    };

    // 初回レンダリング時にサイズを設定
    handleResize();

    // resizeイベントリスナーを設定
    window.addEventListener('resize', handleResize);

    // クリーンアップ関数
    return () => window.removeEventListener('resize', handleResize);
  }, []); // 空の依存配列でマウント時とアンマウント時のみ実行

  return windowSize;
};
