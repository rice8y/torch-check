//! Explicit, ignored smoke test against the official PyTorch wheel index.

use torch_check::index::{IndexOptions, load_index};

#[tokio::test]
#[ignore = "requires network access to download.pytorch.org"]
async fn official_torch_index_refresh_is_complete_and_nonempty() {
    let cache = tempfile::tempdir().expect("temporary cache");
    let options = IndexOptions {
        cache_dir: Some(cache.path().to_path_buf()),
        refresh: true,
        ..IndexOptions::default()
    };

    let loaded = load_index(&options)
        .await
        .expect("official index refresh should succeed");

    assert!(!loaded.snapshot.wheels.is_empty());
    assert!(
        loaded
            .snapshot
            .wheels
            .iter()
            .all(|wheel| wheel.package == "torch")
    );
}
