$(document).ready(function() {
    // Load initial data
    $.get('/get_data', function(data) {
        // Populate states dropdown
        $.each(data.states, function(i, state) {
            $('#state').append($('<option>', {
                value: state,
                text: state
            }));
        });
        
        // Populate seasons dropdown
        $.each(data.seasons, function(i, season) {
            $('#season').append($('<option>', {
                value: season,
                text: season
            }));
        });
        
        // Populate crops dropdown
        $.each(data.crops, function(i, crop) {
            $('#crop').append($('<option>', {
                value: crop,
                text: crop
            }));
        });
    });
    
    // Update districts when state changes
    $('#state').change(function() {
        var selectedState = $(this).val();
        $('#district').empty().append($('<option>', {
            value: '',
            text: 'Select District'
        }));
        
        if (selectedState) {
            $.post('/get_districts', {state: selectedState}, function(data) {
                $.each(data.districts, function(i, district) {
                    $('#district').append($('<option>', {
                        value: district,
                        text: district
                    }));
                });
            });
        }
    });
    
    // Handle form submission
    $('#predictionForm').submit(function(e) {
        e.preventDefault();
        $('#errorAlert').addClass('d-none');
        
        $.post('/predict', $(this).serialize(), function(data) {
            if (data.error) {
                $('#errorAlert').text(data.error).removeClass('d-none');
                $('#resultCard').addClass('d-none');
            } else {
                $('#productionResult').text(data.production);
                $('#resultCard').removeClass('d-none');
            }
        });
    });
});