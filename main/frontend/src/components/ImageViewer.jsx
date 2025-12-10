import React, { useState, useRef, useEffect } from 'react';
import { ZoomIn, ZoomOut, Move, Sun, Contrast, RotateCcw } from 'lucide-react';
import { Button } from './ui/button';
import { Slider } from './ui/slider';

const ImageViewer = ({ src, alt }) => {
  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  
  const containerRef = useRef(null);
  const imageRef = useRef(null);

  const handleZoomIn = () => setScale(prev => Math.min(prev + 0.1, 3));
  const handleZoomOut = () => setScale(prev => Math.max(prev - 0.1, 0.5));
  
  const handleReset = () => {
    setScale(1);
    setPosition({ x: 0, y: 0 });
    setBrightness(100);
    setContrast(100);
  };

  const handleMouseDown = (e) => {
    e.preventDefault();
    setIsDragging(true);
    setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;
    e.preventDefault();
    setPosition({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleWheel = (e) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      const delta = e.deltaY * -0.01;
      const newScale = Math.min(Math.max(scale + delta, 0.5), 3);
      setScale(newScale);
    }
  };

  useEffect(() => {
    const container = containerRef.current;
    if (container) {
      container.addEventListener('wheel', handleWheel, { passive: false });
    }
    return () => {
      if (container) {
        container.removeEventListener('wheel', handleWheel);
      }
    };
  }, [scale]);

  return (
    <div className="flex flex-col gap-4 w-full h-full">
      {/* Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-2 p-2 bg-slate-100 dark:bg-slate-800 rounded-lg">
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="icon" onClick={handleZoomOut} title="Zoom Out">
            <ZoomOut className="h-4 w-4" />
          </Button>
          <span className="text-xs w-12 text-center">{Math.round(scale * 100)}%</span>
          <Button variant="ghost" size="icon" onClick={handleZoomIn} title="Zoom In">
            <ZoomIn className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="icon" onClick={handleReset} title="Reset">
            <RotateCcw className="h-4 w-4" />
          </Button>
        </div>
        
        <div className="flex items-center gap-4 px-2 flex-1 min-w-[200px]">
          <div className="flex items-center gap-2 flex-1">
            <Sun className="h-4 w-4 text-slate-500" />
            <Slider 
              value={[brightness]} 
              min={50} 
              max={150} 
              step={1} 
              onValueChange={(val) => setBrightness(val[0])}
              className="w-full"
            />
          </div>
          <div className="flex items-center gap-2 flex-1">
            <Contrast className="h-4 w-4 text-slate-500" />
            <Slider 
              value={[contrast]} 
              min={50} 
              max={150} 
              step={1} 
              onValueChange={(val) => setContrast(val[0])}
              className="w-full"
            />
          </div>
        </div>
      </div>

      {/* Image Container */}
      <div 
        ref={containerRef}
        className="relative w-full h-[400px] bg-black overflow-hidden rounded-lg cursor-move border border-slate-200 dark:border-slate-700"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <div 
          style={{
            transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
            filter: `brightness(${brightness}%) contrast(${contrast}%)`,
            transition: isDragging ? 'none' : 'transform 0.1s ease-out',
            width: '100%',
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          <img 
            ref={imageRef}
            src={src} 
            alt={alt} 
            className="max-w-full max-h-full object-contain pointer-events-none select-none"
            draggable={false}
          />
        </div>
        
        <div className="absolute bottom-2 right-2 bg-black/50 text-white text-xs px-2 py-1 rounded pointer-events-none">
          Drag to Pan • Scroll to Zoom
        </div>
      </div>
    </div>
  );
};

export default ImageViewer;
