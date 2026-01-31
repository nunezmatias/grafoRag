# ✅ TODO: Library Validation Roadmap

This document outlines the critical tests required to ensure the `graphrag_core` library is fully functional for *new* data ingestion, distinct from the pre-loaded Climate dataset.

## 1. Test `GraphBuilder` with External Data
The core ingestion logic (`builder.py`) was migrated from the original scripts. We need to verify it works correctly within the package structure.

- [ ] **Create a Dummy JSON Dataset**
    - Construct a small, valid JSON file (`test_data.json`) following the schema defined in `docs/TECHNICAL_REPORT.md`.
    - Include at least 3 nodes and 2 edges with text properties.
- [ ] **Run the Builder Pipeline**
    - Execute the build process using the library:
      ```python
      from graphrag_core import GraphBuilder
      builder = GraphBuilder(output_vector_db_path="./test_db")
      builder.build(input_json_path="./test_data.json", output_json_skeleton_path="./test_skeleton.json")
      ```
- [ ] **Verify Output Integrity**
    - Check if `./test_db` (ChromaDB folder) is created and populated.
    - Check if `./test_skeleton.json` contains the correct topology (nodes/edges without heavy text).

## 2. Test Engine with the New Database
Once the test database is built, we must ensure the `GraphRAGEngine` can load and query it effectively.

- [ ] **Initialize Engine with Custom Paths**
    ```python
    from graphrag_core import GraphRAGEngine
    engine = GraphRAGEngine(vector_db_path="./test_db", graph_json_path="./test_skeleton.json")
    ```
- [ ] **Perform a Semantic Search**
    - Run `engine.search("query relevant to test data")` and assert that results are returned.
- [ ] **Verify Metadata Extraction**
    - Confirm that custom metadata fields from `test_data.json` are correctly preserved and retrievable in the search results.

## 3. Deployment & Integration Tests
- [ ] **Colab "Swap Brain" Test:** Upload the `test_data.json` (or a real alternative dataset) to Google Drive, zip it according to specs, and try the `gdrive_id` initialization in a fresh Colab notebook.
- [ ] **Schema Robustness:** Test with a JSON that has missing optional fields (e.g., nodes without references) to ensure the builder doesn't crash.

## 4. Graph Quality Improvements
- [ ] **Improve Skeleton Graph Mapping:** Redo the mapping between the real graph and the skeleton graph, taking into consideration other information that was previously missed.

## 5. Development Workflow Improvements
- [ ] **Investigate Google Colab VS Code Extension:** Analyze the features of the [Google Colab for VS Code](https://marketplace.visualstudio.com/items?itemName=google.colab-vscode) extension. Determine if it allows a smoother hybrid workflow (local editing, cloud execution) and document the setup process for the team.
