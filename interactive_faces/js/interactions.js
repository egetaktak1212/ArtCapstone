const addDragBehavior = (svg) => {
    const drag = d3.drag()
        .on("start", function(event) {
            d3.select(this).raise().attr("stroke", "white").attr("stroke-width", 2);
        })
        .on("drag", function(event) {
            // move the circle
            d3.select(this)
                .attr("cx", event.x)
                .attr("cy", event.y);

            // redraw every line using its two classes to find its circles
        })
        .on("end", function(event) {
            d3.select(this).attr("stroke", null).attr("stroke-width", null);
                        svg.selectAll("line").each(function() {
                const line = d3.select(this);
                const classes = line.attr("class").split(" ")
                const p1id = classes[0];  // e.g. "p-1"
                const p2id = classes[1];  // e.g. "p-2"

                line
                    .attr("x1", svg.select(`#${p1id}`).attr("cx"))
                    .attr("y1", svg.select(`#${p1id}`).attr("cy"))
                    .attr("x2", svg.select(`#${p2id}`).attr("cx"))
                    .attr("y2", svg.select(`#${p2id}`).attr("cy"));
            });
        });

    svg.selectAll("circle").call(drag);
}