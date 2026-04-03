const loadImages_3 = (data) => {

    const img = new Image();
    img.src = '../images/lenny.jpg';
    img.onload = function() {
        const width = this.naturalWidth;
        const height = this.naturalHeight;

        const svg = d3.select("#image3")
            .append("svg")
                .attr("height", height)
                .attr("width", width)

        svg.append("image")
            .attr("href", "../images/lenny.jpg")

        connections.forEach((feature) => {
            feature.forEach((pair) => {
                const p1 = data.find(d => d.landmark_id == pair[0]);  // find by id, not array index
                const p2 = data.find(d => d.landmark_id == pair[1]);

                svg.append("line")
                    .attr("class", `p-${pair[0]} p-${pair[1]}`)
                    .attr("x1", p1.pixel_x)
                    .attr("y1", p1.pixel_y)
                    .attr("x2", p2.pixel_x)
                    .attr("y2", p2.pixel_y)
                    .attr("stroke", "#FF0000")
                    .attr("stroke-width", 2)
            })
        });

        svg.selectAll("circle")
            .data(data)
            .join("circle")
                .attr("id", (d) => `p-${d.landmark_id}`)  // p-1 through p-21
                .attr("r", 4)
                .attr("cx", d => d.pixel_x)
                .attr("cy", d => d.pixel_y)
                .attr("fill", "#FF0000")
                

                addDragBehavior(svg);
            };
}