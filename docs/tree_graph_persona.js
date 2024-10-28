var treeDatas_persona = [
    { data: treeData_neutral, label: "Neutral" },
    { data: treeData_asd, label: "ASD" },
    { data: treeData_age10, label: "10-year-old" },
    { data: treeData_age30, label: "30-year-old" },
    { data: treeData_age70, label: "70-year-old" },
    { data: treeData_female, label: "Female" },
    { data: treeData_male, label: "Male" },
    { data: treeData_disable, label: "Physically-disabled" }, 
    { data: treeData_income_low, label: "Low income" }, 
    { data: treeData_income_high, label: "High income" },
    { data: treeData_education_low, label: "Low education" }, 
    { data: treeData_education_high, label: "High education" }
];

var selector2 = d3.select("#treeSelector_persona");
treeDatas_persona.forEach(function(treeData_persona, index2) {
    selector2.append("option")
        .attr("value", index2)
        .text(treeData_persona.label);
});

var width2 = 1400;
var height2 = 400;

var currentIndex = 0;
d3.select("#chart2").selectAll("*").remove();
drawTree(treeDatas_persona[currentIndex].data);

selector2.on("change", function() {
    currentIndex = +this.value;
    d3.select("#chart2").selectAll("*").remove();
    drawTree(treeDatas_persona[currentIndex].data);
});


function scale(value, minOriginal, maxOriginal, minTarget, maxTarget) {
    return ((value - minOriginal) / (maxOriginal - minOriginal)) * (maxTarget - minTarget) + minTarget;
}

function drawTree(graphData) {

    const xValues = graphData.nodes.map(node => node.x);
    const yValues = graphData.nodes.map(node => node.y);

    const minX = Math.min(...xValues) - 60;
    const maxX = Math.max(...xValues) + 60;
    const minY = Math.min(...yValues) - 60;
    const maxY = Math.max(...yValues) + 60;

    graphData.nodes.forEach(node => {
        node.x = scale(node.x, minX, maxX, 0, width2);
        node.y = scale(node.y, minY, maxY, 0, height2);
    });

    const svg = d3.select("#chart2").append("svg")
        .attr("width", width2)
        .attr("height", height2);

    // Center of the SVG
    const centerX = width2 / 2;
    const centerY = height2 / 2;
    const radius = 150; // Radius of the initial circular layout

    // Sort nodes by color for circular arrangement
    graphData.nodes.sort((a, b) => a.color.localeCompare(b.color));

    // Calculate the angle increment for arranging nodes in a circle
    const angleIncrement = (2 * Math.PI) / graphData.nodes.length;

    // Assign initial circular positions based on sorted colors
    graphData.nodes.forEach((node, index1) => {
        const angle = index1 * angleIncrement;
        node.initialX = centerX + radius * Math.cos(angle);
        node.initialY = centerY + radius * Math.sin(angle);
    });

    const link = svg.selectAll(".link")
        .data(graphData.links)
        .enter().append("line")
        .attr("class", "link")
        .style("opacity", 0.5)
        .style("stroke", "#999")
        .style("stroke-width", 1)
        // Start the links from circular positions
        .attr("x1", d => graphData.nodes.find(node => node.id === d.source).initialX)
        .attr("y1", d => graphData.nodes.find(node => node.id === d.source).initialY)
        .attr("x2", d => graphData.nodes.find(node => node.id === d.target).initialX)
        .attr("y2", d => graphData.nodes.find(node => node.id === d.target).initialY);

    const node = svg.selectAll(".node")
        .data(graphData.nodes)
        .enter().append("g")
        .attr("class", "node")
        // Start nodes at circular positions
        .attr("transform", d => `translate(${d.initialX}, ${d.initialY})`);

    node.append("circle")
        .attr("r", 10)
        .style("opacity", 0.7)
        .style("fill", d => d.color)
        .style("stroke", "none");

    const labels = node.append("text")
        .attr("text-anchor", "middle")
        .attr("dy", ".35em")
        .attr("dx", 0)
        .text(d => d.id)
        .style("font-size", "12px")
        .style("fill", "#333")
        .attr("transform", "rotate(-45)")
        .on("click", function(event, d) {
            highlightEdges(d.id);
        });

    // Function to highlight edges connected to a specific node
    function highlightEdges(nodeId) {
        // Reset all edges to default styling
        link.style("stroke", "#999")
            .style("stroke-width", 3)
            .style("opacity", 0.5);

        // Highlight the edges connected to the clicked node
        link.filter(d => d.source === nodeId || d.target === nodeId)
            .style("stroke", "#f00") // Change to a different color (e.g., red)
            .style("stroke-width", 3) // Increase stroke width for visibility
            .style("opacity", 1);
    }

    // Animate the transition to final positions
    function animateGraph() {
        link.transition()
            .duration(500)
            .attr("x1", d => graphData.nodes.find(node => node.id === d.source).x)
            .attr("y1", d => graphData.nodes.find(node => node.id === d.source).y)
            .attr("x2", d => graphData.nodes.find(node => node.id === d.target).x)
            .attr("y2", d => graphData.nodes.find(node => node.id === d.target).y);

        node.transition()
            .duration(500)
            .attr("transform", d => `translate(${d.x}, ${d.y})`);
    }

    setTimeout(animateGraph, 1000);
}


