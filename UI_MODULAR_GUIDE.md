# Modular UI Structure

The Lagrangian Transport Simulator now uses a modular UI architecture that makes it easy to add, remove, or customize interface sections.

## Directory Structure

```
templates/
├── index.html              # Original monolithic version (backup)
├── index-modular.html      # New modular version
└── sections/               # Modular UI sections
    ├── simulation-controls.html
    ├── parameters.html
    ├── wind-data.html
    └── emission-hotspots.html

static/
├── css/
│   └── styles.css         # Extracted CSS styles
└── js/
    ├── main.js            # Main application logic
    ├── section-loader.js  # Dynamic section loading utility
    └── ui-config.js       # UI section configuration
```

## How It Works

1. **Section Files**: Each UI section is a standalone HTML file in `templates/sections/`
2. **Configuration**: Sections are registered in `static/js/ui-config.js`
3. **Dynamic Loading**: `section-loader.js` loads sections at runtime
4. **Main Logic**: All JavaScript functions are in `static/js/main.js`

## Adding a New Section

### Step 1: Create the Section HTML

Create a new file in `templates/sections/`, for example `advanced-settings.html`:

```html
<!-- Advanced Settings -->
<div class="control-section">
    <h3>⚙️ Advanced Settings</h3>
    
    <div class="param-control">
        <label>
            Custom Parameter:
            <span class="param-value" id="custom-param-val">100</span>
        </label>
        <input type="range" id="custom-param" min="0" max="200" value="100" 
               oninput="updateCustomParam(this.value)">
    </div>
</div>
```

### Step 2: Register in Configuration

Add your section to `static/js/ui-config.js`:

```javascript
const UI_SECTIONS = [
    // ... existing sections ...
    {
        name: 'Advanced Settings',
        file: '/sections/advanced-settings.html',
        container: 'controls-container',
        enabled: true  // Set to false to disable
    }
];
```

### Step 3: Add JavaScript Functions (if needed)

If your section needs JavaScript functions, add them to `static/js/main.js`:

```javascript
async function updateCustomParam(value) {
    document.getElementById('custom-param-val').textContent = value;
    await fetch('/api/custom-param', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({custom_param: parseFloat(value)})
    });
}
```

### Step 4: Test

1. Restart the Flask server
2. Open the browser
3. Your new section should appear in the controls panel

## Section Guidelines

### HTML Structure

- Wrap section content in `<div class="control-section">`
- Use `<h3>` for section titles
- Use existing CSS classes for consistency:
  - `.param-control` - for parameter controls
  - `.param-value` - for displaying values
  - `.button` - for buttons
  - `.tooltip` and `.help-icon` - for help tooltips

### JavaScript Functions

- Functions should be async when making API calls
- Update UI immediately for responsive feel
- Use `console.log()` for debugging
- Handle errors gracefully

### CSS Styling

- Use CSS custom properties for theming (e.g., `var(--text-color)`)
- Follow existing patterns for dark mode support
- Test in both light and dark modes

## Switching Between Versions

### Use Modular Version (Recommended)

In `app.py`, ensure the route points to the modular version:

```python
@app.route('/')
def index():
    return render_template('index-modular.html')
```

### Use Original Version

If you need to revert:

```python
@app.route('/')
def index():
    return render_template('index.html')
```

## Benefits of Modular Structure

1. **Easier Maintenance**: Each section is isolated
2. **Better Collaboration**: Multiple developers can work on different sections
3. **Reusability**: Sections can be reused in other projects
4. **Conditional Loading**: Enable/disable sections via configuration
5. **Clean Code**: Separates concerns (HTML, CSS, JS)
6. **Version Control**: Smaller, focused commits

## Troubleshooting

### Section Not Loading

1. Check browser console for errors
2. Verify file path in `ui-config.js` is correct
3. Ensure Flask route `/sections/<filename>` is working
4. Check that section is `enabled: true` in config

### Styling Issues

1. Verify CSS classes match existing patterns
2. Check `static/css/styles.css` is loaded
3. Test dark mode compatibility

### JavaScript Errors

1. Check function names match between HTML and JS
2. Verify API endpoints exist in backend
3. Use browser DevTools to debug

## Example: Complete New Section

Here's a complete example of adding a "Physics Options" section:

**File: `templates/sections/physics-options.html`**
```html
<div class="control-section">
    <h3>🔬 Physics Options</h3>
    
    <div class="param-control">
        <label>
            <input type="checkbox" id="enable-gravity" onchange="toggleGravity()">
            Enable Gravity Effects
        </label>
    </div>
    
    <div class="param-control">
        <label>
            Particle Density: <span class="param-value" id="density-val">1.0</span> kg/m³
        </label>
        <input type="range" id="particle-density" min="0.1" max="5.0" step="0.1" value="1.0"
               oninput="updateDensity(this.value)">
    </div>
</div>
```

**Add to `static/js/ui-config.js`:**
```javascript
{
    name: 'Physics Options',
    file: '/sections/physics-options.html',
    container: 'controls-container',
    enabled: true
}
```

**Add to `static/js/main.js`:**
```javascript
async function toggleGravity() {
    const enabled = document.getElementById('enable-gravity').checked;
    await fetch('/api/physics/gravity', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({enabled: enabled})
    });
}

async function updateDensity(value) {
    document.getElementById('density-val').textContent = parseFloat(value).toFixed(1);
    await fetch('/api/physics/density', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({density: parseFloat(value)})
    });
}
```

That's it! Your new section is now integrated into the UI.
