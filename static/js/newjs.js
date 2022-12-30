$(document).ready(function () {

    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    function readURL(input){
        if(input.files && input.files[0])
        {
            var reader = new FileReader();
            reader.onload = function(e){
                $('#imagePreview').attr('src',e.target.result);
            }
            reader.readAsDataURL(input.files[0])
        }
    }

    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data=new FormData($('#upload-file')[0]);
        // Show loading animation
        $(this).hide();
        $('.loader').show();
        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data){
               // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);                                
                $('#result').text(' Result: ' + data);
                if(data == "Yes Brain Tumor"){
                    $('#result').css({"color": "red", "filter": "drop-shadow(0 0 20px red) drop-shadow(0 0 60px red)","animation":"None"});
                }else{
                    $('#result').css({"color": "green","filter": "drop-shadow(0 0 20px green) drop-shadow(0 0 60px green)","animation":"None"});
                }
                //console.log(data)
                console.log('Success!');
            },
        });
    });

});