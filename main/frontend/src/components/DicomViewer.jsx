import React, { useEffect, useRef, useState } from 'react';
import cornerstone from 'cornerstone-core';
import cornerstoneWADOImageLoader from 'cornerstone-wado-image-loader';
import dicomParser from 'dicom-parser';
import Hammer from 'hammerjs';
import * as cornerstoneMath from 'cornerstone-math';
import * as cornerstoneTools from 'cornerstone-tools';

// Initialize external dependencies
cornerstoneWADOImageLoader.external.cornerstone = cornerstone;
cornerstoneWADOImageLoader.external.dicomParser = dicomParser;
cornerstoneTools.external.cornerstone = cornerstone;
cornerstoneTools.external.Hammer = Hammer;
cornerstoneTools.external.cornerstoneMath = cornerstoneMath;

// Initialize tools
cornerstoneTools.init();

const DicomViewer = ({ file }) => {
    const elementRef = useRef(null);
    const [imageId, setImageId] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (!file) return;

        try {
            const imageId = cornerstoneWADOImageLoader.wadouri.fileManager.add(file);
            setImageId(imageId);

            const element = elementRef.current;
            cornerstone.enable(element);

            cornerstone.loadImage(imageId).then(image => {
                cornerstone.displayImage(element, image);
                
                // Add tools
                const WwwcTool = cornerstoneTools.WwwcTool;
                const ZoomTool = cornerstoneTools.ZoomTool;
                const PanTool = cornerstoneTools.PanTool;
                const ZoomMouseWheelTool = cornerstoneTools.ZoomMouseWheelTool;

                cornerstoneTools.addTool(WwwcTool);
                cornerstoneTools.addTool(ZoomTool);
                cornerstoneTools.addTool(PanTool);
                cornerstoneTools.addTool(ZoomMouseWheelTool);

                // Active tools
                cornerstoneTools.setToolActive('Wwwc', { mouseButtonMask: 1 }); // Left Click
                cornerstoneTools.setToolActive('Pan', { mouseButtonMask: 2 });  // Right Click
                cornerstoneTools.setToolActive('Zoom', { mouseButtonMask: 4 }); // Middle Click
                cornerstoneTools.setToolActive('ZoomMouseWheel', {});

            }).catch(err => {
                console.error("Error loading DICOM:", err);
                setError("Failed to load DICOM image. " + err.message);
            });

            return () => {
                cornerstone.disable(element);
            };
        } catch (err) {
            console.error("Error initializing viewer:", err);
            setError("Failed to initialize viewer.");
        }
    }, [file]);

    if (error) {
        return <div className="text-red-500 p-4 border border-red-500 rounded">{error}</div>;
    }

    return (
        <div className="relative w-full h-[500px] bg-black border border-gray-700 rounded-lg overflow-hidden">
            <div 
                ref={elementRef} 
                style={{ width: '100%', height: '100%' }}
                onContextMenu={(e) => e.preventDefault()} // Prevent right-click menu
            />
            <div className="absolute top-2 left-2 text-white text-xs bg-black/50 p-1 rounded pointer-events-none">
                Left: Window/Level | Right: Pan | Middle/Wheel: Zoom
            </div>
        </div>
    );
};

export default DicomViewer;
