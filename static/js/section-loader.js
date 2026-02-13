/**
 * Section Loader - Dynamically loads HTML sections into the controls panel
 */

const SectionLoader = {
    /**
     * Load a section from an HTML file and insert it into a container
     * @param {string} sectionFile - Path to the section HTML file
     * @param {string} containerId - ID of the container element
     */
    async loadSection(sectionFile, containerId) {
        try {
            const response = await fetch(sectionFile);
            if (!response.ok) {
                console.error(`Failed to load section: ${sectionFile}`);
                return false;
            }
            
            const html = await response.text();
            const container = document.getElementById(containerId);
            if (container) {
                container.insertAdjacentHTML('beforeend', html);
                return true;
            }
            
            console.error(`Container not found: ${containerId}`);
            return false;
        } catch (error) {
            console.error(`Error loading section ${sectionFile}:`, error);
            return false;
        }
    },

    /**
     * Load multiple sections in order
     * @param {Array} sections - Array of {file, container} objects
     */
    async loadSections(sections) {
        for (const section of sections) {
            await this.loadSection(section.file, section.container);
        }
        console.log('✓ All sections loaded');
    }
};

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SectionLoader;
}
