def test_bundle_skips_gpulayer_home_and_run_outputs(tmp_path):
    import io
    import zipfile

    from gpulayer.artifacts.store import build_source_bundle

    (tmp_path / "train.py").write_text("print(1)")
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "metadata.db").write_bytes(b"x")
    (tmp_path / "state" / "artifacts").mkdir()
    (tmp_path / "state" / "artifacts" / "big").write_bytes(b"0" * 1000)
    (tmp_path / "gpulayer_runs" / "job-1").mkdir(parents=True)
    (tmp_path / "gpulayer_runs" / "job-1" / "last.ckpt").write_bytes(b"w")
    names = zipfile.ZipFile(io.BytesIO(build_source_bundle(tmp_path))).namelist()
    assert names == ["train.py"]
