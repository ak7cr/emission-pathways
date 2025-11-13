# Help Tooltips Feature - Implementation Summary

## ✅ Feature Added: Interactive Parameter Help

Added **help icons (?)** next to every parameter with detailed hover tooltips.

---

## 📋 Parameters with Help Tooltips

### 1. 🗺️ Simulation Scale
- **What it does**: Controls total area covered by simulation
- **Auto-scales**: Grid resolution, hotspot positions, turbulent diffusion
- **Examples**:
  - Block (5km): Street-level pollution
  - City (50km): Urban air quality
  - Regional (200km): Multi-city analysis

### 2. 🚀 Speed Multiplier
- **What it does**: Multiplies base time step for faster/slower simulation
- **Effect**: Higher values = faster particle movement per frame
- **Examples**:
  - 0.1x: Slow-motion analysis
  - 1.0x: Real-time (default)
  - 10x: Fast forward for long runs

### 3. 🌪️ Turbulent Diffusion (σ_turb)
- **What it does**: Controls random particle spreading from turbulence
- **Effect**: Higher values = wider, more dispersed plumes
- **Auto-scaling**: Scales automatically with domain size
- **Examples**:
  - 0.5-1.5 m/s: Rural/calm conditions
  - 2-3 m/s: Urban typical (default)
  - 5-10 m/s: Strong convection/mixing

### 4. 💨 Wind Speed Multiplier
- **What it does**: Scales wind velocity to simulate different weather
- **Effect**: How fast pollutants transport downwind
- **Examples**:
  - 0.1x: Calm conditions (~1-2 m/s)
  - 1.0x: Normal winds (~12 m/s)
  - 5.0x: Storm/hurricane (~60 m/s)

### 5. 👥 Particles per Hotspot (npph)
- **What it does**: Number of particles from each emission source
- **Trade-off**: More particles = smoother but slower
- **Examples**:
  - 500: Fast, rough visualization
  - 2500: Balanced (default)
  - 10000: Smooth, publication-quality

### 6. ⏱️ Base Time Step (dt)
- **What it does**: Base duration of each simulation step
- **Note**: Multiplied by speed multiplier for actual dt
- **Examples**:
  - 10s: High accuracy, slow
  - 30s: Balanced (default)
  - 120s: Fast, less accurate

### 7. 🌬️ Wind Field Type
- **What it does**: Choose between synthetic or real wind data
- **Options**:
  - Synthetic: Testing, demonstrations
  - Real (ERA5/GFS): Actual case studies

### 8. 👁️ Show Wind Vectors
- **What it does**: Toggle wind direction arrows on map
- **Tip**: Turn off for cleaner concentration visualizations

---

## 🎨 Tooltip Features

### Visual Design
- **Icon**: Blue circle with white "?" 
- **Hover effect**: Slightly enlarges on hover
- **Tooltip box**: Dark background (adapts to theme)
- **Width**: 280px for comfortable reading
- **Animation**: Smooth fade-in on hover
- **Arrow**: Points down to the help icon

### Content Structure
Each tooltip contains:
1. **Title** (bold, colored): Parameter name
2. **Description**: What it controls
3. **Examples section**: Practical values with 💡 icon
4. **Italic text**: For tips and recommendations

### Theme Support
- **Light mode**: Dark tooltip on light background
- **Dark mode**: Darker tooltip with colored border
- **Both modes**: High contrast for readability

---

## 💻 Technical Implementation

### CSS Classes Added
```css
.help-icon          - The ? icon itself
.tooltip            - Wrapper for icon + tooltip text
.tooltiptext        - The popup tooltip content
.example            - Example values section
```

### How It Works
1. Hover over "?" icon
2. CSS `:hover` triggers visibility
3. Tooltip fades in above the icon
4. Move mouse away - tooltip fades out

### Files Modified
- `templates/index.html`:
  - Added 88 lines of CSS for tooltips
  - Updated 8 parameter labels with help icons
  - Each tooltip has custom content

---

## 🚀 Usage

### For Users
1. Look for blue "?" icons next to parameters
2. Hover mouse over any "?" 
3. Read explanation and examples
4. Use recommended values as starting points

### For Developers
To add more tooltips:
```html
<span class="tooltip">
    <span class="help-icon">?</span>
    <span class="tooltiptext">
        <strong>Parameter Name</strong>
        Description here.
        <span class="example">💡 Examples:<br>
        • Value 1: Use case<br>
        • Value 2: Use case</span>
    </span>
</span>
```

---

## ✨ Benefits

1. **Self-documenting UI**: No need to read external docs
2. **Contextual help**: Information right where you need it
3. **Learning tool**: Users understand physics behind parameters
4. **Best practices**: Examples guide proper usage
5. **Accessibility**: Hover-based, no clicking required

---

## 🎯 Next Steps

1. **Restart server**: `python app.py`
2. **Test tooltips**: Hover over each "?" icon
3. **Check both themes**: Toggle dark mode
4. **Verify examples**: Try recommended values

---

**Status**: ✅ Complete and ready to use!
