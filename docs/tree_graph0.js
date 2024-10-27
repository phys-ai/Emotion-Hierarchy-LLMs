      var treeData = {
        "name": "Root",
        "children": [
          { 
            "name": "Child 1",
            "children": [
              { "name": "Child 1.1" },
              { "name": "Child 1.2" }
            ]
          },
          { 
            "name": "Child 2",
            "children": [
              { "name": "Child 2.1" },
              { 
                "name": "Child 2.2",
                "children": [
                  { "name": "Child 2.2.1" },
                  { "name": "Child 2.2.2" }
                ]
              }
            ]
          }
        ]
      };


var node_colors = {
    "love": "rgba(254, 224, 139, 1.0)",
    "adoration": "rgba(254, 224, 139, 1.0)",
    "affection": "rgba(254, 226, 143, 1.0)",
    "fondness": "rgba(254, 228, 147, 1.0)",
    "liking": "rgba(254, 230, 149, 1.0)",
    "attraction": "rgba(254, 232, 153, 1.0)",
    "caring": "rgba(254, 234, 157, 1.0)",
    "tenderness": "rgba(254, 237, 161, 1.0)",
    "compassion": "rgba(254, 239, 165, 1.0)",
    "sentimentality": "rgba(254, 242, 169, 1.0)",
    "arousal": "rgba(254, 243, 171, 1.0)",
    "desire": "rgba(254, 245, 175, 1.0)",
    "lust": "rgba(254, 248, 179, 1.0)",
    "passion": "rgba(254, 250, 183, 1.0)",
    "infatuation": "rgba(254, 253, 187, 1.0)",
    "longing": "rgba(254, 254, 190, 1.0)",
    "joy": "rgba(244, 109, 67, 1.0)",
    "amusement": "rgba(244, 109, 67, 1.0)",
    "bliss": "rgba(244, 111, 68, 1.0)",
    "cheerfulness": "rgba(244, 111, 68, 1.0)",
    "gaiety": "rgba(244, 114, 69, 1.0)",
    "glee": "rgba(245, 116, 70, 1.0)",
    "jolliness": "rgba(245, 119, 71, 1.0)",
    "joviality": "rgba(245, 121, 72, 1.0)",
    "delight": "rgba(245, 121, 72, 1.0)",
    "enjoyment": "rgba(246, 124, 74, 1.0)",
    "gladness": "rgba(246, 126, 75, 1.0)",
    "happiness": "rgba(246, 129, 76, 1.0)",
    "jubilation": "rgba(247, 131, 77, 1.0)",
    "elation": "rgba(247, 134, 78, 1.0)",
    "satisfaction": "rgba(247, 134, 78, 1.0)",
    "ecstasy": "rgba(247, 137, 79, 1.0)",
    "euphoria": "rgba(248, 139, 81, 1.0)",
    "enthusiasm": "rgba(248, 142, 82, 1.0)",
    "zeal": "rgba(248, 144, 83, 1.0)",
    "zest": "rgba(249, 147, 84, 1.0)",
    "excitement": "rgba(249, 147, 84, 1.0)",
    "thrill": "rgba(249, 149, 85, 1.0)",
    "exhilaration": "rgba(250, 152, 86, 1.0)",
    "contentment": "rgba(250, 154, 88, 1.0)",
    "pleasure": "rgba(250, 157, 89, 1.0)",
    "pride": "rgba(251, 159, 90, 1.0)",
    "triumph": "rgba(251, 159, 90, 1.0)",
    "eagerness": "rgba(251, 162, 91, 1.0)",
    "hope": "rgba(251, 165, 92, 1.0)",
    "optimism": "rgba(252, 167, 94, 1.0)",
    "enthrallment": "rgba(252, 170, 95, 1.0)",
    "rapture": "rgba(252, 170, 95, 1.0)",
    "relief": "rgba(252, 172, 96, 1.0)",
    "surprise": "rgba(254, 254, 190, 1.0)",
    "amazement": "rgba(254, 254, 190, 1.0)",
    "astonishment": "rgba(230, 245, 152, 1.0)",
    "anger": "rgba(158, 1, 66, 1.0)",
    "aggravation": "rgba(158, 1, 66, 1.0)",
    "irritation": "rgba(158, 1, 66, 1.0)",
    "agitation": "rgba(160, 3, 66, 1.0)",
    "annoyance": "rgba(162, 5, 67, 1.0)",
    "grouchiness": "rgba(164, 8, 67, 1.0)",
    "grumpiness": "rgba(166, 10, 68, 1.0)",
    "exasperation": "rgba(168, 12, 68, 1.0)",
    "frustration": "rgba(170, 15, 69, 1.0)",
    "rage": "rgba(173, 17, 69, 1.0)",
    "outrage": "rgba(175, 20, 70, 1.0)",
    "fury": "rgba(177, 22, 70, 1.0)",
    "wrath": "rgba(179, 24, 71, 1.0)",
    "hostility": "rgba(181, 27, 71, 1.0)",
    "ferocity": "rgba(183, 29, 72, 1.0)",
    "bitterness": "rgba(186, 32, 72, 1.0)",
    "hate": "rgba(188, 34, 73, 1.0)",
    "loathing": "rgba(190, 36, 73, 1.0)",
    "scorn": "rgba(192, 39, 74, 1.0)",
    "spite": "rgba(194, 41, 74, 1.0)",
    "vengefulness": "rgba(196, 44, 75, 1.0)",
    "dislike": "rgba(196, 44, 75, 1.0)",
    "resentment": "rgba(198, 46, 75, 1.0)",
    "disgust": "rgba(201, 48, 76, 1.0)",
    "revulsion": "rgba(203, 51, 76, 1.0)",
    "contempt": "rgba(205, 53, 77, 1.0)",
    "envy": "rgba(207, 56, 77, 1.0)",
    "jealousy": "rgba(209, 58, 78, 1.0)",
    "torment": "rgba(211, 60, 78, 1.0)",
    "sadness": "rgba(102, 194, 165, 1.0)",
    "agony": "rgba(102, 194, 165, 1.0)",
    "suffering": "rgba(99, 191, 165, 1.0)",
    "hurt": "rgba(97, 189, 166, 1.0)",
    "anguish": "rgba(97, 189, 166, 1.0)",
    "depression": "rgba(95, 187, 167, 1.0)",
    "despair": "rgba(93, 184, 168, 1.0)",
    "hopelessness": "rgba(91, 182, 169, 1.0)",
    "gloom": "rgba(91, 182, 169, 1.0)",
    "glumness": "rgba(89, 180, 170, 1.0)",
    "unhappiness": "rgba(87, 178, 171, 1.0)",
    "grief": "rgba(85, 175, 172, 1.0)",
    "sorrow": "rgba(85, 175, 172, 1.0)",
    "woe": "rgba(83, 173, 173, 1.0)",
    "misery": "rgba(81, 171, 174, 1.0)",
    "melancholy": "rgba(79, 168, 175, 1.0)",
    "dismay": "rgba(79, 168, 175, 1.0)",
    "disappointment": "rgba(77, 166, 176, 1.0)",
    "displeasure": "rgba(75, 164, 177, 1.0)",
    "guilt": "rgba(75, 164, 177, 1.0)",
    "shame": "rgba(73, 162, 178, 1.0)",
    "regret": "rgba(71, 159, 179, 1.0)",
    "remorse": "rgba(69, 157, 180, 1.0)",
    "alienation": "rgba(69, 157, 180, 1.0)",
    "isolation": "rgba(67, 155, 181, 1.0)",
    "neglect": "rgba(65, 153, 181, 1.0)",
    "loneliness": "rgba(63, 150, 182, 1.0)",
    "rejection": "rgba(63, 150, 182, 1.0)",
    "homesickness": "rgba(61, 148, 183, 1.0)",
    "defeat": "rgba(59, 146, 184, 1.0)",
    "dejection": "rgba(57, 143, 185, 1.0)",
    "insecurity": "rgba(57, 143, 185, 1.0)",
    "embarrassment": "rgba(55, 141, 186, 1.0)",
    "humiliation": "rgba(53, 139, 187, 1.0)",
    "insult": "rgba(53, 139, 187, 1.0)",
    "pity": "rgba(51, 137, 188, 1.0)",
    "sympathy": "rgba(50, 134, 188, 1.0)",
    "fear": "rgba(50, 134, 188, 1.0)",
    "alarm": "rgba(50, 134, 188, 1.0)",
    "shock": "rgba(54, 130, 186, 1.0)",
    "fright": "rgba(56, 128, 185, 1.0)",
    "horror": "rgba(59, 123, 183, 1.0)",
    "terror": "rgba(62, 119, 181, 1.0)",
    "panic": "rgba(64, 117, 180, 1.0)",
    "hysteria": "rgba(68, 112, 177, 1.0)",
    "mortification": "rgba(71, 108, 175, 1.0)",
    "anxiety": "rgba(75, 103, 173, 1.0)",
    "nervousness": "rgba(76, 101, 172, 1.0)",
    "tenseness": "rgba(80, 96, 170, 1.0)",
    "uneasiness": "rgba(83, 92, 168, 1.0)",
    "apprehension": "rgba(85, 90, 167, 1.0)",
    "worry": "rgba(88, 85, 165, 1.0)",
    "distress": "rgba(92, 81, 163, 1.0)",
    "dread": "rgba(94, 79, 162, 1.0)"
};


var treeData= {'name': ' happiness', 'children': [{'name': ' hope', 'children': [{'name': ' optimism', 'children': []}]}, {'name': ' excitement', 'children': [{'name': ' hope', 'children': [{'name': ' optimism', 'children': []}]}, {'name': ' thrill', 'children': []}, {'name': ' enthusiasm', 'children': []}, {'name': ' desire', 'children': []}, {'name': ' passion', 'children': []}]}, {'name': ' amusement', 'children': []}, {'name': ' pride', 'children': [{'name': ' triumph', 'children': []}, {'name': ' passion', 'children': []}]}, {'name': ' relief', 'children': [{'name': ' loneliness', 'children': [{'name': ' isolation', 'children': []}]}]}, {'name': ' delight', 'children': []}, {'name': ' satisfaction', 'children': [{'name': ' pleasure', 'children': []}]}, {'name': ' enjoyment', 'children': []}, {'name': ' pleasure', 'children': []}, {'name': ' bliss', 'children': []}, {'name': ' affection', 'children': []}, {'name': ' triumph', 'children': []}, {'name': ' caring', 'children': []}]}

var margin = { top: 40, right: 10, bottom: 40, left: 10 },
    width = 1500 - margin.right - margin.left,
    height = 1400 - margin.top - margin.bottom;

var i = 0,
    duration = 750;

// Change the tree layout size to be horizontal
var tree = d3.layout.tree()
    .size([height, width])
    .separation(function(a, b) { return (a.parent === b.parent ? 1 : 2); });

// Change projection to swap x and y for a horizontal layout
var diagonal = d3.svg.diagonal()
    .projection(function(d) { return [d.x, d.y]; });

var svg = d3.select("#chart0").append("svg")
    .attr("width", width + margin.right + margin.left)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

var root = treeData;

update(root);
function update(source) {
  var nodes = tree.nodes(root).reverse(),
      links = tree.links(nodes);
  nodes.forEach(function(d) { d.y = d.depth * 80; });
  var node = svg.selectAll("g.node")
      .data(nodes, function(d) { return d.id || (d.id = ++i); });
  var nodeEnter = node.enter().append("g")
      .attr("class", "node")
      .attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });
  nodeEnter.append("circle")
      .attr("r", 10)
      .style("fill", function(d) { return node_colors[d.name] || "#ccc"; })
      .style("fill-opacity", 0.6)
      .style("stroke", "none");
  nodeEnter.append("text")
      .attr("x", function(d) { return d.children || d._children ? -13 : 13; })
      .attr("dy", ".75em")
      .attr("text-anchor", function(d) { return d.children || d._children ? "end" : "start"; })
      .text(function(d) { return d.name; })
      .style("fill-opacity", 1);

  var link = svg.selectAll("path.link")
      .data(links, function(d) { return d.target.id; });
  link.enter().insert("path", "g")
      .attr("class", "link")
      .attr("d", diagonal);
}

