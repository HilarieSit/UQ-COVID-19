$(document).ready(function() {
    $("form").submit(function(e) {
      $("#loadingtxt").show();
    });

    $('.fa-info-circle').hover(function() {
      $('.overlay').show().delay(1000).fadeOut();
    },
    function () {
      $('.overlay').fadeOut();
    });

    $("#fileupload").change(function(e) {
      $("#previewimg").css('display', 'inline-block');
      $('#previewimg').attr('src', '../static/uploads/'+e.target.files[0].name);
      $("form").css('text-align', 'left');
      $("#submitlabel").css('display', 'inline-block');
    });
});
