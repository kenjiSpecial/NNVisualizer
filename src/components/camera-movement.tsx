import { useFrame, useThree } from '@react-three/fiber';
import { useRef } from 'react';

export function CameraMovement() {
  const { camera } = useThree();

  const time = useRef(0);

  useFrame(() => {
    time.current += 0.01;
    const radius = 5;
    camera.position.x = radius * Math.sin(time.current);
    camera.position.z = radius * Math.cos(time.current);
    camera.lookAt(0, 0, 0);
  });

  return null;
}
