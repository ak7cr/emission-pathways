/**
 * Configuration for UI sections
 * Add new sections here to include them in the interface
 */

const UI_SECTIONS = [
    {
        name: 'Model Selector',
        file: '/sections/model-selector.html',
        container: 'controls-container',
        enabled: true,
        order: 0
    },
    {
        name: 'Simulation Controls',
        file: '/sections/simulation-controls.html',
        container: 'controls-container',
        enabled: true,
        order: 1
    },
    {
        name: 'Parameters',
        file: '/sections/parameters.html',
        container: 'controls-container',
        enabled: true,
        order: 2
    },
    {
        name: 'Wind Data',
        file: '/sections/wind-data.html',
        container: 'controls-container',
        enabled: true,
        order: 3
    },
    {
        name: 'Emission Hotspots',
        file: '/sections/emission-hotspots.html',
        container: 'controls-container',
        enabled: true,
        order: 4
    }
    // Add more sections here as needed
    // Example:
    // {
    //     name: 'Advanced Settings',
    //     file: '/sections/advanced-settings.html',
    //     container: 'controls-container',
    //     enabled: false  // Set to true to enable
    // }
];
