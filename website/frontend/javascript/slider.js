document.addEventListener("DOMContentLoaded", function () {
    // Initialize the Bootstrap carousel
    const myCarousel = document.querySelector('#customCarousel1');
  
    if (myCarousel) {
      const carousel = new bootstrap.Carousel(myCarousel, {
        interval: 5000, // Time between slides (ms)
        ride: 'carousel', // Automatically start the carousel
        pause: false, // Keeps sliding even when hovered
        wrap: true // Allows cycling back to the first slide
      });
    }
  });
  