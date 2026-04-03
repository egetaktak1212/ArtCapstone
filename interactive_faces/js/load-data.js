d3.csv("../data/cameron _landmarks.csv", d3.autoType).then(data => {
  loadImages(data);

  // createTooltip();
  // handleMouseEvents();
});


d3.csv("../data/Walter_landmarks.csv", d3.autoType).then(data => {
  console.log(data)
  loadImages_1(data);
});

d3.csv("../data/lenny_landmarks.csv", d3.autoType).then(data => {
  console.log(data)
  loadImages_3(data);
});
