import json

import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="HTML Tooltip Viewer", layout="wide")

st.title("Interactive HTML Content with Tooltips, Line Drawing and PNG Export")


default_html = """\
<div class="container">
  <p class="intro-text">You are <strong>one of Jane's mutual friends on Facebook</strong>.</p>
  <div class="content-box">
    <p class="content-text">
      <strong>👩 <span class="hover-tooltip dashed-underline">Jane<span class="tooltip">Your friend,<br/> a transgender woman.</span></span> is</strong> sharing
      <span class="highlight-blue">reasons for the end of her marriage</span>
      with <strong>her friends (including you)</strong> by posting a Facebook post.
    </p>
  </div>
</div>
"""


def extract_tooltips(html: str) -> list[dict[str, str | int]]:
    """Extract tooltip contents from spans with class='tooltip'."""
    tooltips = []
    remaining_html = html
    tooltip_index = 0

    markers = (
        'class="tooltip">',
        "class='tooltip'>",
    )

    while True:
        marker = next(
            (item for item in markers if item in remaining_html),
            None,
        )

        if marker is None:
            break

        start = remaining_html.find(marker) + len(marker)
        end = remaining_html.find("</span>", start)

        if end == -1:
            break

        tooltip_text = (
            remaining_html[start:end]
            .replace("<br/>", " ")
            .replace("<br>", " ")
            .strip()
        )

        if tooltip_text:
            tooltips.append(
                {
                    "content": tooltip_text,
                    "id": f"tooltip-{tooltip_index}",
                    "index": tooltip_index,
                }
            )
            tooltip_index += 1

        remaining_html = remaining_html[end + len("</span>") :]

    return tooltips


def build_complete_html(
    html_input: str,
    tooltip_width: int,
    download_filename: str,
) -> tuple[str, list[dict[str, str | int]], bool]:
    tooltips = extract_tooltips(html_input)
    has_tooltips = bool(tooltips)

    tooltip_html = "".join(
        f"""
        <div
          class="tooltip-box"
          id="{tooltip['id']}"
          style="
            top: {50 + int(tooltip['index']) * 10}px;
            right: {20 + int(tooltip['index']) * 10}px;
          "
        >
          <div>{tooltip['content']}</div>
        </div>
        """
        for tooltip in tooltips
    )

    streamlit_data = json.dumps(
        {
            "filename": download_filename,
            "hasTooltips": has_tooltips,
            "tooltips": tooltips,
        }
    )

    complete_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body {{
    margin: 0;
    padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }}

  .container {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    width: 50%;
    padding-top: 30px;
    padding-right: 200px;
    padding-left: 200px;
    padding-bottom: 60px;
    border-radius: 8px;
    border: 1px solid #e0e0e0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    position: relative;
    background-color: white;
  }}

  .intro-text {{
    font-size: 16px;
    color: #2c3e50;
    margin-bottom: 20px;
  }}

  .content-box {{
    background-color: #f1f3f4;
    padding: 20px;
    border-radius: 8px;
    position: relative;
  }}

  .hover-tooltip {{
    position: relative;
    cursor: default;
    padding: 2px 4px;
    border-radius: 1px;
  }}

  .hover-tooltip .tooltip {{
    display: none !important;
  }}

  .highlight-blue {{
    color: #3498db;
    font-weight: 600;
  }}

  .dashed-underline {{
    text-decoration: none;
    position: relative;
  }}

  .dashed-underline::after {{
    content: '';
    position: absolute;
    left: 0;
    bottom: 0;
    width: 100%;
    height: 1px;
    background-image: linear-gradient(to right, #000 40%, transparent 40%);
    background-size: 4px 1px;
    background-repeat: repeat-x;
  }}

  .content-text {{
    font-size: 16px;
    color: #34495e;
    margin: 0;
    line-height: 1.6;
  }}

  .tooltip-box {{
    position: absolute;
    width: {tooltip_width}px;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 14px;
    font-style: italic;
    color: #6c757d;
    cursor: move;
    z-index: 1000;
    user-select: none;
    transition: background-color 0.2s ease;
    word-wrap: break-word;
  }}

  .tooltip-box:hover {{
    border-color: #adb5bd;
  }}

  .tooltip-box.dragging {{
    transform: rotate(2deg);
  }}

  .dashed-line {{
    stroke: #666666;
    stroke-width: 1;
    stroke-dasharray: 1,1;
    fill: none;
    pointer-events: none;
  }}

  .line-canvas {{
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 500;
    overflow: visible;
  }}

  .main-container {{
    position: relative;
  }}

  .drawing-mode {{
    cursor: crosshair !important;
  }}

  .drawing-mode * {{
    cursor: crosshair !important;
  }}

  .line-controls {{
    background: rgba(255, 255, 255, 0.95);
    padding: 10px 15px;
    border-radius: 6px;
    border: 1px solid #dee2e6;
    font-size: 14px;
    margin-bottom: 15px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }}

  .line-btn {{
    background: #007bff;
    color: white;
    border: none;
    padding: 6px 12px;
    border-radius: 4px;
    cursor: pointer;
    margin-right: 8px;
    font-size: 13px;
  }}

  .line-btn:hover {{
    background: #0056b3;
  }}

  .line-btn.active {{
    background: #28a745;
  }}

  .download-btn {{
    position: absolute;
    top: 10px;
    right: 10px;
    z-index: 2000;
    background: #28a745;
    color: white;
    border: none;
    padding: 8px 12px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
  }}

  .download-btn:hover {{
    background: #218838;
  }}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
</head>
<body>

<button class="download-btn" id="downloadBtn">💾 Download PNG</button>

<div class="line-controls">
  <button class="line-btn" id="drawLineBtn">Draw Line</button>
  <button class="line-btn" id="clearLinesBtn">Clear Lines</button>
  <span id="lineStatus">Click to start drawing</span>
</div>

<div class="main-container" id="main-container">
  <svg class="line-canvas" id="lineCanvas"></svg>

  {html_input}

  {tooltip_html if has_tooltips else ""}
</div>

<script>
window.streamlitData = {streamlit_data};

(function() {{
    let isDrawingMode = false;
    let firstPoint = null;
    let lineCount = 0;

    const container = document.getElementById("main-container");
    const lineCanvas = document.getElementById("lineCanvas");
    const drawLineBtn = document.getElementById("drawLineBtn");
    const clearLinesBtn = document.getElementById("clearLinesBtn");
    const lineStatus = document.getElementById("lineStatus");
    const downloadBtn = document.getElementById("downloadBtn");

    const hasTooltips = window.streamlitData.hasTooltips;
    const tooltipBoxes = [];

    if (hasTooltips) {{
        window.streamlitData.tooltips.forEach(tooltip => {{
            const element = document.getElementById(tooltip.id);
            if (element) {{
                tooltipBoxes.push({{
                    element: element,
                    id: tooltip.id,
                    isDragging: false,
                    currentX: 0,
                    currentY: 0,
                    initialX: 0,
                    initialY: 0,
                    xOffset: 0,
                    yOffset: 0
                }});
            }}
        }});
    }}

    function dragStart(e) {{
        if (isDrawingMode || !hasTooltips) return;

        e.preventDefault();
        const clientX = e.type.includes('touch') ? e.touches[0].clientX : e.clientX;
        const clientY = e.type.includes('touch') ? e.touches[0].clientY : e.clientY;

        const targetTooltip = tooltipBoxes.find(tooltip =>
            e.target === tooltip.element || tooltip.element.contains(e.target)
        );

        if (targetTooltip) {{
            targetTooltip.initialX = clientX - targetTooltip.xOffset;
            targetTooltip.initialY = clientY - targetTooltip.yOffset;
            targetTooltip.isDragging = true;
            targetTooltip.element.classList.add('dragging');
        }}
    }}

    function dragEnd(e) {{
        tooltipBoxes.forEach(tooltip => {{
            if (tooltip.isDragging) {{
                tooltip.isDragging = false;
                tooltip.element.classList.remove('dragging');
                tooltip.initialX = tooltip.currentX;
                tooltip.initialY = tooltip.currentY;
            }}
        }});
    }}

    function drag(e) {{
        e.preventDefault();
        const clientX = e.type.includes('touch') ? e.touches[0].clientX : e.clientX;
        const clientY = e.type.includes('touch') ? e.touches[0].clientY : e.clientY;

        tooltipBoxes.forEach(tooltip => {{
            if (tooltip.isDragging) {{
                tooltip.currentX = clientX - tooltip.initialX;
                tooltip.currentY = clientY - tooltip.initialY;
                tooltip.xOffset = tooltip.currentX;
                tooltip.yOffset = tooltip.currentY;
                tooltip.element.style.transform =
                    `translate(${{tooltip.currentX}}px, ${{tooltip.currentY}}px)`;
            }}
        }});
    }}

    function toggleDrawingMode() {{
        isDrawingMode = !isDrawingMode;
        firstPoint = null;

        if (isDrawingMode) {{
            container.classList.add('drawing-mode');
            drawLineBtn.classList.add('active');
            drawLineBtn.textContent = 'Exit Drawing';
            lineStatus.textContent = 'Click first point';
        }} else {{
            container.classList.remove('drawing-mode');
            drawLineBtn.classList.remove('active');
            drawLineBtn.textContent = 'Draw Line';
            lineStatus.textContent = 'Click to start drawing';
        }}
    }}

    function handleCanvasClick(e) {{
        if (!isDrawingMode) return;

        if (e.target.closest('.tooltip-box')) {{
            return;
        }}

        if (!container.contains(e.target) && e.target !== container) {{
            return;
        }}

        e.preventDefault();
        e.stopPropagation();

        const containerRect = container.getBoundingClientRect();
        const x = e.clientX - containerRect.left;
        const y = e.clientY - containerRect.top;

        if (!firstPoint) {{
            firstPoint = {{ x, y }};
            lineStatus.textContent = 'Click second point';
        }} else {{
            drawLine(firstPoint.x, firstPoint.y, x, y);
            firstPoint = null;
            lineStatus.textContent = 'Click first point';
        }}
    }}

    function drawLine(x1, y1, x2, y2) {{
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', x1);
        line.setAttribute('y1', y1);
        line.setAttribute('x2', x2);
        line.setAttribute('y2', y2);
        line.setAttribute('class', 'dashed-line');
        line.setAttribute('id', `line-${{++lineCount}}`);
        lineCanvas.appendChild(line);
    }}

    function clearAllLines() {{
        lineCanvas.innerHTML = '';
        lineCount = 0;
        firstPoint = null;
        if (isDrawingMode) {{
            lineStatus.textContent = 'Click first point';
        }}
    }}

    function downloadScreenshot() {{
        downloadBtn.textContent = '📸 Capturing...';
        downloadBtn.disabled = true;

        html2canvas(container, {{
            scale: 2,
            backgroundColor: null,
            useCORS: true,
            allowTaint: true,
            logging: false
        }}).then(canvas => {{
            const link = document.createElement('a');
            link.download = window.streamlitData.filename;
            link.href = canvas.toDataURL('image/png');

            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            downloadBtn.textContent = '💾 Download PNG';
            downloadBtn.disabled = false;
        }}).catch(error => {{
            console.error('Screenshot failed:', error);
            alert('Screenshot failed. Please try again.');
            downloadBtn.textContent = '💾 Download PNG';
            downloadBtn.disabled = false;
        }});
    }}

    if (hasTooltips && tooltipBoxes.length > 0) {{
        tooltipBoxes.forEach(tooltip => {{
            tooltip.element.addEventListener("mousedown", dragStart, false);
            tooltip.element.addEventListener("touchstart", dragStart, false);
            tooltip.element.addEventListener("selectstart", function(e) {{
                e.preventDefault();
            }});
        }});
    }}

    document.addEventListener("mousemove", drag, false);
    document.addEventListener("mouseup", dragEnd, false);
    document.addEventListener("touchmove", drag, false);
    document.addEventListener("touchend", dragEnd, false);

    drawLineBtn.addEventListener('click', function(e) {{
        e.stopPropagation();
        toggleDrawingMode();
    }});

    clearLinesBtn.addEventListener('click', function(e) {{
        e.stopPropagation();
        clearAllLines();
    }});

    downloadBtn.addEventListener('click', function(e) {{
        e.stopPropagation();
        downloadScreenshot();
    }});

    container.addEventListener('click', handleCanvasClick);
}})();
</script>

</body>
</html>
"""

    return complete_html, tooltips, has_tooltips


st.markdown("### Manual Input Mode")

tooltip_width = st.slider(
    "Tooltip Box Width (px)",
    min_value=100,
    max_value=300,
    value=100,
    step=10,
)

html_input = st.text_area(
    "HTML Content:",
    value=default_html,
    height=200,
    help=(
        "Enter HTML content using the container structure. "
        "Tooltips should use the hover-tooltip class."
    ),
)

download_filename = st.text_input(
    "PNG filename",
    value="tooltip_view.png",
)

if not download_filename.lower().endswith(".png"):
    download_filename += ".png"

if html_input and html_input.strip():
    complete_html, tooltips, has_tooltips = build_complete_html(
        html_input,
        tooltip_width,
        download_filename,
    )

    st.markdown(
        """
        ### Instructions:
        - **Edit the HTML content** in the text area above to customize the display
        - **Adjust tooltip width** using the slider above (if tooltip content exists)
        - **Click and drag the tooltip box** to move it around freely (if visible)
        - **Click "Draw Line"** to enter drawing mode, then click two points to create a dashed line
        - **Click "Clear Lines"** to remove all drawn lines
        - **Click "💾 Download PNG"** to capture and download a high-quality screenshot
        """
    )

    components.html(complete_html, height=500, scrolling=False)

    st.info(f"📁 Screenshot will be saved as: **{download_filename}**")
    if has_tooltips:
        st.success(
            f"✅ Found {len(tooltips)} tooltip(s) - all displayed and draggable"
        )
    else:
        st.warning(
            "⚠️ No tooltip content found - tooltip boxes hidden, "
            "but screenshot still available"
        )
else:
    st.warning("Please enter some HTML content to display.")
