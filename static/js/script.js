document.addEventListener('DOMContentLoaded', function() {
    // File upload validation
    const fileInput = document.querySelector('input[type="file"]');
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                // Check file size (max 10MB)
                if (file.size > 10 * 1024 * 1024) {
                    alert('File is too large! Maximum size is 10MB.');
                    this.value = '';
                }
                
                // Check file type
                const validTypes = ['image/jpeg', 'image/png', 'image/heic'];
                if (!validTypes.includes(file.type)) {
                    alert('Only JPG, PNG, or HEIC files are allowed!');
                    this.value = '';
                }
            }
        });
    }
    
    // Display filename when selected
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const label = this.nextElementSibling;
            if (this.files.length > 0) {
                label.textContent = this.files[0].name;
                label.style.borderColor = '#28a745';
                label.style.backgroundColor = '#e6f7e6';
            } else {
                label.textContent = 'Choose an image';
                label.style.borderColor = '';
                label.style.backgroundColor = '';
            }
        });
    }
});