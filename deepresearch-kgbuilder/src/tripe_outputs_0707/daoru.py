from py2neo import Graph, Node, Relationship
import os
import json

# === Neo4j 配置 ===
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "a786547789"
JSONL_FOLDER = "/Users/yangjianxin/Downloads/open_deep_research-main/src/tripe_outputs_0707"

# === 连接 Neo4j ===
graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# === 创建 Disease 父节点 ===
parent_disease = Node("DiseaseGroup", name="Disease")
graph.merge(parent_disease, "DiseaseGroup", "name")

# === 遍历 JSONL 文件导入 ===
for file in os.listdir(JSONL_FOLDER):
    if not file.endswith(".jsonl"):
        continue
    print(f"📥 导入文件: {file}")
    filepath = os.path.join(JSONL_FOLDER, file)

    with open(filepath, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"❌ JSON 解码错误（{file}, 第 {lineno} 行）: {e}")
                continue

            if obj["type"] == "entity":
                name = obj["entity_text"]
                label = obj["entity_type"]
                attr = obj.get("attr", {})
                attr["reference"] = obj.get("reference", "Unknown")
                attr["name"] = name  # 主键属性

                # 创建实体节点
                node = Node(label, **attr)
                graph.merge(node, label, "name")

                # 如果是 Disease，创建包含关系
                if label == "Disease":
                    disease_node = graph.nodes.match("Disease", name=name).first()
                    if disease_node:
                        rel = Relationship(parent_disease, "INCLUDES", disease_node)
                        graph.merge(rel)

            elif obj["type"] == "relation":
                head = obj["head"]
                tail = obj["tail"]
                rel_type = obj["relation_type"].upper()

                node_a = graph.nodes.match(name=head).first()
                node_b = graph.nodes.match(name=tail).first()

                if node_a and node_b:
                    rel = Relationship(node_a, rel_type, node_b)
                    graph.merge(rel)

print("✅ 所有数据已成功导入 Neo4j 图谱！")
