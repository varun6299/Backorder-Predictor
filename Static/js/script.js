// const allLinks = document.querySelectorAll('a:link');
// allLinks.forEach((function(link) {
//  link.addEventListener('click',function(e){
//    e.preventDefault();
//    const href = link.getAttribute('href');

    // scroll back to stopped
//    if (href === "#") window.scrollTo({
//      top: 0,
//      behavior:"smooth",
//    });
//  });
//});


$('a[href*="#"]')
  // Remove links that don't actually link to anything
  .not('[href="#"]')
  .not('[href="#0"]')
  .click(function(event) {
    // On-page links
    if (
      location.pathname.replace(/^\//, '') == this.pathname.replace(/^\//, '')
      &&
      location.hostname == this.hostname
    ) {
      // Figure out element to scroll to
      var target = $(this.hash);
      target = target.length ? target : $('[name=' + this.hash.slice(1) + ']');
      // Does a scroll target exist?
      if (target.length) {
        // Only prevent default if animation is actually gonna happen
        event.preventDefault();
        $('html, body').animate({
          scrollTop: target.offset().top
        }, 1000, function() {
          // Callback after animation
          // Must change focus!
          var $target = $(target);
          $target.focus();
          if ($target.is(":focus")) { // Checking if the target was focused
            return false;
          } else {
            $target.attr('tabindex','-1'); // Adding tabindex for elements not focusable
            $target.focus(); // Set focus again
          };
        });
      }
    }
  });


  $(document).ready(function() {

   /* For the sticky navigation */
   $('.js--section-how').waypoint(function(direction) {
   if (direction == "down") {
   $('header').addClass('sticky');
   } else {
   $('header').removeClass('sticky');
   }
   });
   });
