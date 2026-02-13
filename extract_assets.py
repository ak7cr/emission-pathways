#!/usr/bin/env python3
"""
Extract CSS and JavaScript from index.html to separate files
"""

# Read index.html
with open('templates/index.html', 'r') as f:
    content = f.read()

# Extract CSS (between <style> and </style>)
css_start = content.find('<style>') + 7
css_end = content.find('</style>')
css_content = content[css_start:css_end].strip()

# Extract JavaScript (between <script> and </script>, last occurrence)
js_start = content.rfind('<script>') + 8
js_end = content.rfind('</script>')
js_content = content[js_start:js_end].strip()

# Write CSS file
with open('static/css/styles.css', 'w') as f:
    f.write(css_content)
print('✓ Created static/css/styles.css')

# Write JavaScript file
with open('static/js/main.js', 'w') as f:
    f.write(js_content)
print('✓ Created static/js/main.js')

print('\n✓ Asset extraction complete!')
print('Files created:')
print('  - static/css/styles.css')
print('  - static/js/main.js')
