"use client"

import { useState, useEffect, useRef } from "react"
import "./PDFPolygonOverlay.css"

const PDFPolygonOverlay = ({ docData, currentPage, scale, containerRef, onPolygonClick, zoomLevel, pageHeight, activeElementId }) => {
  const [activePolygon, setActivePolygon] = useState(null)
  const [tooltip, setTooltip] = useState({ visible: false, content: "", x: 0, y: 0 })
  const overlayRef = useRef(null)

  // Reset active polygon when page changes
  // Update activePolygon when activeElementId or page changes
  useEffect(() => {
    setActivePolygon(activeElementId || null)
    setTooltip({ visible: false, content: "", x: 0, y: 0 })
  }, [activeElementId])

  // Get elements for all pages with page number extracted from doc_id
  const getCurrentPageElements = () => {
    if (!docData || !Array.isArray(docData)) {
      console.log("[Overlay] docData is missing or not an array:", docData);
      return [];
    }

    // Map all elements from docData
    const allElements = docData
      .map((item) => {
        const docId = item?.doc_id;
        const polygon = item?.polygon;

        // Validate docId and polygon
        if (!docId || typeof docId !== "string" || !polygon || !Array.isArray(polygon)) {
          console.warn("[Overlay] Invalid item, missing doc_id or polygon:", item);
          return null;
        }

        // Extract page number from doc_id (e.g., /page/9/ListItem/10 -> 10)
        const match = docId.match(/^\/page\/(\d+)\//);
        const pageNumber = match && match[1] ? parseInt(match[1]) + 1 : null;

        if (!pageNumber) {
          console.warn(`[Overlay] Invalid doc_id format, cannot extract page number: ${docId}`);
          return null;
        }

        return {
          id: docId,
          polygon: polygon,
          pageNumber: pageNumber
        };
      })
      .filter((item) => item !== null); // Remove invalid items

    if (allElements.length === 0) {
      console.warn("[Overlay] No valid elements found in docData");
    }

    return allElements;
  };


  const handlePolygonClick = (element, event) => {
    event.stopPropagation()

    // Toggle active state
    setActivePolygon(activePolygon === element.id ? null : element.id)

    // Hide tooltip
    setTooltip({ ...tooltip, visible: false })

    // Call the parent callback
    if (onPolygonClick) {
      onPolygonClick(element)
    }
  }

  const handlePolygonMouseEnter = (element, event) => {
    const rect = event.currentTarget.getBoundingClientRect()
    const containerRect = containerRef.current.getBoundingClientRect()

    // Calculate position for tooltip
    const x = rect.left + rect.width / 2 - containerRect.left
    let y = rect.top - 10 - containerRect.top

    // Determine tooltip direction
    let direction = "bottom"
    if (y < 40) {
      y = rect.bottom + 10 - containerRect.top
      direction = "top"
    }

    setTooltip({
      visible: true,
      content: element.html || element.block_type,
      x,
      y,
      direction,
    })
  }

  const handlePolygonMouseLeave = () => {
    setTooltip({ ...tooltip, visible: false })
  }

  // Convert polygon coordinates to CSS style
  const getPolygonStyle = (polygon, pageNumber) => {
    if (!polygon || polygon.length < 3) return {}

    // Calculate bounding box
    let minX = Number.POSITIVE_INFINITY,
      minY = Number.POSITIVE_INFINITY,
      maxX = Number.NEGATIVE_INFINITY,
      maxY = Number.NEGATIVE_INFINITY

    polygon.forEach((point) => {
      minX = Math.min(minX, point[0])
      minY = Math.min(minY, point[1])
      maxX = Math.max(maxX, point[0])
      maxY = Math.max(maxY, point[1])
    })

    const sca_x = 1.25
    const sca_y = 1.24
    
    const sca_wh = 1.25

    // console.log(pageHeight)

    // Apply scale factor using the passed 'scale' prop
    return {
      left: `${5 + minX * sca_x * scale}px`,
      top: `${pageHeight * (pageNumber-1) * 1.24 + minY * sca_y * scale}px`,
      width: `${(maxX - minX) * sca_wh * scale}px`,
      height: `${(maxY - minY) * sca_wh * scale}px`,
    }
  }


  const elements = getCurrentPageElements()

  return (
    <div className="polygon-overlay-container" ref={overlayRef}>
      {elements.map((element) => (
        <div
          key={element.id}
          className={`polygon-element ${activePolygon === element.id ? "active" : ""}`}
          style={{
            ...getPolygonStyle(element.polygon, element.pageNumber),
            // Background color is now handled by CSS classes
          }}
          onClick={(e) => handlePolygonClick(element, e)}
          onMouseEnter={(e) => handlePolygonMouseEnter(element, e)}
          onMouseLeave={handlePolygonMouseLeave}
          data-element-id={element.id}
          // data-block-type={element.block_type}
        />
      ))}

      {/* {tooltip.visible && (
        <div
          className="polygon-tooltip"
          style={{
            left: `${tooltip.x}px`,
            top: `${tooltip.y}px`,
            transform: tooltip.direction === "top" ? "translateX(-50%)" : "translate(-50%, -100%)",
          }}
        >
          <div className="polygon-tooltip-content" dangerouslySetInnerHTML={{ __html: tooltip.content }} />
          <div className={`polygon-tooltip-arrow ${tooltip.direction}`} />
        </div>
      )} */}
    </div>
  )
}

export default PDFPolygonOverlay
