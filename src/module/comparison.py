from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import torch


def compare_hd_vs_2d_predictions(model, dataset, episode_idx=0, query_sample_id=None,
                                  n_way=2, k_shot=3, query_per_class=3, device='cpu',
                                  random_seed=42):
    """
    Compare model predictions in high-dimensional embedding space vs 2D visualization space.

    This function demonstrates how dimensionality reduction (t-SNE/UMAP) can distort distance
    relationships, making predictions appear incorrect in 2D plots even when the model
    predicts correctly in the original high-dimensional space.

    Args:
        model: The trained PrototypicalNetwork model
        dataset: The dataset to sample from (typically test_dataset)
        episode_idx: Which episode to analyze (default: 0 for first episode)
        query_sample_id: Specific query sample ID to focus on (None = analyze all)
        n_way: Number of classes per episode
        k_shot: Number of support examples per class
        query_per_class: Number of query examples per class
        device: Device to run inference on
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary containing analysis results with distance information
    """
    # Get embedding dimension from model
    embedding_dim = model.final_embedding_dim

    print("=" * 80)
    print(f"ANALYSIS: High-Dimensional ({embedding_dim}D) vs 2D Space Comparison")
    print("=" * 80)
    print(f"Episode: {episode_idx + 1} | Embedding Architecture: {model.embedding_dims}")
    print("This demonstrates how dimensionality reduction can distort distance relationships")
    print("=" * 80)

    # Set random seed for reproducibility
    np.random.seed(random_seed + episode_idx)
    torch.manual_seed(random_seed + episode_idx)

    # Build class-to-indices mapping
    class_to_indices = {}
    for idx in range(len(dataset)):
        _, label, _, _ = dataset[idx]
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(idx)

    classes = list(class_to_indices.keys())

    # Sample episode
    selected_classes = np.random.choice(classes, n_way, replace=False)

    support_indices = []
    query_indices = []

    for cls in selected_classes:
        cls_indices = class_to_indices[cls].copy()
        sampled_indices = np.random.choice(
            cls_indices,
            k_shot + query_per_class,
            replace=False
        )
        support_indices.extend(sampled_indices[:k_shot])
        query_indices.extend(sampled_indices[k_shot:])

    # Get data
    support_data = [dataset[i] for i in support_indices]
    query_data = [dataset[i] for i in query_indices]

    support_images = torch.stack([x[0] for x in support_data]).to(device)
    support_labels = torch.tensor([x[1] for x in support_data])
    query_images = torch.stack([x[0] for x in query_data]).to(device)
    query_labels = torch.tensor([x[1] for x in query_data])

    print(f"\nEpisode {episode_idx + 1} Configuration:")
    print(f"  Selected classes: {selected_classes}")
    print(f"  Support indices: {support_indices}")
    print(f"  Query indices: {query_indices}")

    # Extract embeddings in high-dimensional space
    model.eval()
    with torch.no_grad():
        support_embeddings_hd = model.extract_features(support_images)
        query_embeddings_hd = model.extract_features(query_images)

    # Compute prototypes in high-dimensional space
    support_embeddings_hd_reshaped = support_embeddings_hd.view(n_way, k_shot, -1)
    prototypes_hd = support_embeddings_hd_reshaped.mean(dim=1)

    # Get predictions in high-dimensional space (what model actually uses)
    dists_hd = torch.cdist(query_embeddings_hd, prototypes_hd)
    query_predictions_hd = torch.argmin(dists_hd, dim=1)

    # Apply dimensionality reduction to 2D
    all_embeddings_hd = torch.cat([
        support_embeddings_hd,
        query_embeddings_hd
    ], dim=0).cpu().numpy()

    reducer = TSNE(n_components=2, random_state=random_seed,
                   perplexity=min(30, all_embeddings_hd.shape[0] - 1))
    coords_2d = reducer.fit_transform(all_embeddings_hd)

    # Split back
    n_support = len(support_embeddings_hd)
    n_query = len(query_embeddings_hd)

    support_coords_2d = coords_2d[:n_support]
    query_coords_2d = coords_2d[n_support:n_support + n_query]

    # Compute prototypes in 2D space
    support_coords_2d_reshaped = support_coords_2d.reshape(n_way, k_shot, 2)
    prototypes_2d = support_coords_2d_reshaped.mean(axis=1)

    # Compute distances in 2D space
    dists_2d = np.linalg.norm(
        query_coords_2d[:, np.newaxis, :] - prototypes_2d[np.newaxis, :, :],
        axis=2
    )
    query_predictions_2d = np.argmin(dists_2d, axis=1)

    # Convert class indices to names
    class_names = {0: 'BK', 1: 'FK'}

    # Prepare results
    results = {
        'episode_idx': episode_idx,
        'embedding_dim': embedding_dim,
        'selected_classes': selected_classes,
        'query_samples': []
    }

    print("\n" + "=" * 80)
    print(f"COMPARISON FOR ALL QUERY SAMPLES (Episode {episode_idx + 1})")
    print("=" * 80)

    # Analyze all query samples
    for i in range(len(query_indices)):
        sample_idx = query_indices[i]
        true_label = query_labels[i].item()
        pred_hd = query_predictions_hd[i].item()
        pred_2d = query_predictions_2d[i]

        # Get distances in HD space
        dists_to_centroids_hd = {
            class_names[j]: dists_hd[i, j].item()
            for j in range(n_way)
        }

        # Get distances in 2D space
        dists_to_centroids_2d = {
            class_names[j]: dists_2d[i, j]
            for j in range(n_way)
        }

        sample_result = {
            'sample_idx': sample_idx,
            'sample_id': sample_idx + 1,
            'true_label': class_names[true_label],
            'pred_hd': class_names[pred_hd],
            'pred_2d': class_names[pred_2d],
            'dists_hd': dists_to_centroids_hd,
            'dists_2d': dists_to_centroids_2d,
            'correct_hd': pred_hd == true_label,
            'mismatch': pred_hd != pred_2d
        }
        results['query_samples'].append(sample_result)

        # Print detailed analysis
        print(f"\n{'─' * 80}")
        print(f"Query Sample #{i+1} (Dataset Index: {sample_idx}, Sample ID: {sample_idx + 1})")
        print(f"{'─' * 80}")
        print(f"  True Label: {class_names[true_label]}")
        print(f"\n  HIGH-DIMENSIONAL ({embedding_dim}D) SPACE - Where Model Actually Predicts:")
        for cls_name, dist in dists_to_centroids_hd.items():
            print(f"    Distance to {cls_name} centroid: {dist:.6f}")
        print(f"    Predicted Label: {class_names[pred_hd]} {'CORRECT' if pred_hd == true_label else 'WRONG'}")
        print(f"\n  2D VISUALIZATION SPACE - What You See in Plot:")
        for cls_name, dist in dists_to_centroids_2d.items():
            print(f"    Distance to {cls_name} centroid: {dist:.6f}")
        print(f"    Visual Prediction: {class_names[pred_2d]} {'(matches HD)' if pred_2d == pred_hd else '(DIFFERENT from HD!)'}")

        # Highlight discrepancies
        if pred_hd != pred_2d:
            print(f"\n  WARNING: Dimensionality reduction changed the nearest centroid!")
            print(f"      In {embedding_dim}D: closer to {class_names[pred_hd]}")
            print(f"      In 2D: closer to {class_names[pred_2d]}")

    # Focus on specific query sample if requested
    if query_sample_id is not None:
        print("\n" + "=" * 80)
        print(f"FOCUS ON SAMPLE ID {query_sample_id}")
        print("=" * 80)

        # Find the sample in query list
        target_idx = None
        for i, idx in enumerate(query_indices):
            if idx + 1 == query_sample_id:
                target_idx = i
                break

        if target_idx is not None:
            sample_result = results['query_samples'][target_idx]

            print(f"\n  Sample ID {query_sample_id} Analysis:")
            print(f"   True Label: {sample_result['true_label']}")
            print(f"\n   In {embedding_dim}D Space (Model's Decision):")
            for cls_name, dist in sample_result['dists_hd'].items():
                print(f"     Distance to {cls_name}: {dist:.6f}")
            print(f"     Closer to: {sample_result['pred_hd']}")
            print(f"     Model predicts: {sample_result['pred_hd']} {'CORRECT' if sample_result['correct_hd'] else 'WRONG'}")
            print(f"\n   In 2D Space (What Plot Shows):")
            for cls_name, dist in sample_result['dists_2d'].items():
                print(f"     Distance to {cls_name}: {dist:.6f}")
            print(f"     Visually closer to: {sample_result['pred_2d']}")

            if sample_result['mismatch']:
                print(f"\n   KEY INSIGHT:")
                print(f"      The model predicted this sample as {sample_result['pred_hd']} in {embedding_dim}D space,")
                print(f"      but in the 2D plot it appears closer to {sample_result['pred_2d']}!")
                print(f"      This is why the visualization looks 'non-sense' - it's a")
                print(f"      dimensionality reduction artifact, not a model error.")
            else:
                print(f"\n   In this case, {embedding_dim}D and 2D predictions agree.")
        else:
            print(f"\n  Sample ID {query_sample_id} was not found in Episode {episode_idx + 1} query set.")
            print("   The episode may have sampled different images.")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    total_queries = len(results['query_samples'])
    mismatches = sum(1 for s in results['query_samples'] if s['mismatch'])
    correct_hd = sum(1 for s in results['query_samples'] if s['correct_hd'])

    print(f"  Total Query Samples: {total_queries}")
    print(f"  Model Accuracy in {embedding_dim}D: {correct_hd}/{total_queries} ({100*correct_hd/total_queries:.1f}%)")
    print(f"  HD vs 2D Mismatches: {mismatches}/{total_queries} ({100*mismatches/total_queries:.1f}%)")

    # Create distance table
    print("\n" + "=" * 80)
    print("DISTANCE TABLE: Query Samples to Centroids")
    print("=" * 80)

    table_data = []
    for sample in results['query_samples']:
        row = {
            'Sample_ID': sample['sample_id'],
            'True_Label': sample['true_label'],
            'Pred_Label': sample['pred_hd'],
            f'CBK-{embedding_dim}D': sample['dists_hd']['BK'],
            'CBK-2D': sample['dists_2d']['BK'],
            f'CFK-{embedding_dim}D': sample['dists_hd']['FK'],
            'CFK-2D': sample['dists_2d']['FK'],
        }
        table_data.append(row)

    df_distances = pd.DataFrame(table_data)

    # Display the table
    print("\n" + df_distances.to_string(index=False))

    # Add table to results
    results['distance_table'] = df_distances

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("The 2D visualization is useful for understanding clustering patterns,")
    print(f"but should NOT be used to judge individual prediction correctness.")
    print(f"The model makes decisions in {embedding_dim}D space, where distances can be very different!")
    print("=" * 80)

    return results
