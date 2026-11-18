# Edge Face Gateway — v0.4.2

Docker-first CPU runtime for multi-camera face detection and capture. v0.4.2 retains the v0.4.1 Web camera management, encrypted RTSP source maintenance, bounded stream testing and dynamic per-camera Worker lifecycle on top of the v0.2.0 SQLite/Alembic persistence foundation. The v0.1.3 detection/capture pipeline and YOLOv12n-face model remain unchanged. HailoRT/DFC is still excluded until Hailo hardware is available.

## 1. Camera streams

Each camera is configured independently. Dual-stream cameras use `sub` for detection and timestamp-matched `main` frames for full-resolution capture; single-stream cameras use `main` for both. The main-stream ring stores JPEG samples at configurable quality (`main_stream_ring_jpeg_quality`, default 96). Timestamps remain host receive timestamps, not camera RTP/PTS timestamps.

## 2. Captures and face crops

A detection is mapped from detector-frame coordinates to the selected source-frame dimensions. The gateway then:

- saves the full source frame as `*_frame.jpg`;
- saves the same source frame with green detection rectangles and confidence labels as `*_detections.jpg`;
- evaluates each face on its original-resolution face region before context padding;
- saves only faces that pass the quality gates as `*_face_NNN_score_0.xxx.jpg`;
- keeps the original and annotated full frames as diagnostic evidence, including rejected detections (rejected boxes are red);
- records accepted/rejected counts, quality metrics and rejection reasons in `QualityEvent` / `CaptureManifest` records.

Face crops are **image crops only**. The gateway does not perform identity recognition, feature/embedding extraction, person matching, tracking or biometric template creation. These crops can be used as source images for a later, separately designed person-comparison workflow; their suitability depends on camera resolution, focus, motion blur, face size and capture alignment.

Relevant `capture` settings in `config/cameras.yaml`:

| Setting | Default | Purpose |
|---|---:|---|
| `jpeg_quality` | 96 | Original and annotated frame JPEG quality |
| `face_jpeg_quality` | 98 | Face crop JPEG quality |
| `face_crop_padding_ratio` | 0.25 | Context padding relative to face-box width/height |
| `main_stream_ring_jpeg_quality` | 96 | JPEG quality in the dual-stream main-frame ring |
| `main_stream_ring_sample_interval_ms` | 200 | Main-frame ring sampling interval |

JPEG quality is a compression setting, not a guarantee of optical detail. A crop cannot recover detail absent from the source stream. The dual-stream ring is JPEG-compressed, so the saved frame/crops inherit that compression.

### Generic quality assessment

The gateway does not control shutter, gain, IR illumination or vendor-specific camera settings. It evaluates the received image using face pixel dimensions, Laplacian sharpness (a blur proxy), brightness, contrast and a simple noise estimate. Face size, sharpness and exposure are hard gates; a high score cannot override a failed hard gate. Rejected face crops are not added to the face-crop album, but full-frame and annotated diagnostic images remain available. `QualityEvent` records contain the measured values and rejection reasons.

Initial thresholds are deliberately generic starting points, not validated universal values. Calibrate `capture.quality_assessment` with representative day/night footage from each deployment. A single-frame sharpness metric cannot reliably distinguish motion blur from defocus, poor optics or compression; the current release flags low sharpness without claiming to identify its physical cause. Best-frame selection across time is deferred to v0.1.4.

| Setting | Default | Purpose |
|---|---:|---|
| `min_face_width_px` / `min_face_height_px` | 48 | Minimum face-box dimensions in the selected capture frame |
| `min_sharpness` | 25 | Minimum Laplacian variance |
| `min_brightness` / `max_brightness` | 18 / 238 | Mean grayscale brightness bounds |
| `max_noise` | 14 | Maximum robust noise estimate from locally flat pixels |
| `min_score` | 45 | Minimum weighted quality score |

## 3. Dashboard

- Live monitoring: `http://localhost:8080/`
- Camera management: `http://localhost:8080/cameras`
- Album: `http://localhost:8080/album`
- Event log: `http://localhost:8080/logs`
- Status JSON: `http://localhost:8080/api/status`
- Capture API: `/api/captures?camera_id=camera-01&kind=face`
- Event API: `/api/events?event_type=SaveEvent`
- Live MJPEG: `/stream/{camera_id}`

The album supports camera filtering, image-kind filtering (original frame, annotated frame, face crop), lazy image loading and pagination. Click **打开原图** to inspect the full JPEG.

The event log supports internal diagnostic records and the six MQTT business event types (`camera.online`, `camera.offline`, `capture.created`, `capture.saved`, `upload.complete`, `upload.error`). S3 upload is available when configured; upload completion/failure events are emitted only for actual upload attempts.

JPEG files remain under `data/{camera_id}/`; image metadata and business events are persisted in SQLite. Existing v0.1.3 JPEGs and `logs/runtime.jsonl` are indexed/imported once during the first v0.2.0 startup. The legacy `logs/` directory is mounted read-only only for this import.

## 4. CPU and memory metrics

The dashboard distinguishes:

- process CPU normalized to a 0–100% whole-host scale;
- process CPU core-equivalents (for example, 2.91 means about 2.91 logical cores in use);
- whole-host CPU percentage;
- process RSS memory.

The raw process CPU reading can exceed 100% because `psutil` reports multi-core process utilization. Uptime now uses a monotonic clock to avoid wall-clock adjustments.

## 5. Setup

Requirements: Docker Desktop / Docker Engine with Linux containers and Docker Compose v2.

```powershell
Copy-Item .env.example .env
# Edit .env and set RTSP addresses.
docker compose build
docker compose up -d
```

Default model is `yolov12n-face.onnx`. Set `FACE_MODEL_PATH=/app/models/source/yolov12s-face.onnx` in `.env` to switch models.

## 6. Validation

Run unit tests inside the image:

```powershell
docker compose run --rm --entrypoint pytest face-gateway -q
```

A successful unit test run does not establish camera alignment, correct model preprocessing, face-crop image quality, Docker build success or long-duration stability. Validate using actual camera footage, especially dual-stream box alignment and the saved crop dimensions. This development environment does not have the Docker daemon, live camera streams or the target Hailo hardware, so those deployment checks must be performed on the user's host.

## 7. Scope

Included: RTSP reconnect, CPU ONNX inference, single/dual stream handling, timestamp matching, full-frame and annotated-frame capture, high-resolution face crops, quality assessment and admission control, album UI, SQLite business events, S3-compatible object storage, MQTT event publishing and durable MQTT Outbox, and system metrics. Not included: identity recognition, embeddings, person matching, tracking, Home Assistant MQTT Discovery/inbound control, Hailo HEF runtime or 24-hour acceptance evidence.

## 8. v0.2.0 persistence foundation

v0.2.0 introduces SQLite-backed persistence while keeping the v0.1.3 detection and capture pipeline intact.

### Database and camera configuration

- SQLite database: `data/database/gateway.db` (override with `FACE_DATABASE_PATH`).
- Schema migrations are managed by Alembic and run at startup.
- Existing `config/cameras.yaml` camera entries are imported into SQLite when the camera table is empty. RTSP URLs are encrypted using Fernet; the key is stored separately at `data/secrets/camera.key` (override with `FACE_SECRET_PATH`, or provide `FACE_SECRET_KEY`).
- After camera rows exist, SQLite is the runtime source of truth. Changes to YAML camera entries are not reloaded at runtime. In v0.4.0, camera records and video sources are maintained from `/cameras`; changes are persisted to SQLite and applied to only the affected camera Worker.
- Keep the database, secret key and any `database/pending-logs.jsonl` spool together in protected persistent storage and include them in backups. Losing the key makes encrypted RTSP URLs unrecoverable.

### Images, events and application logs

- JPEG files remain on the local filesystem under `data/{camera_id}/`.
- Each saved original frame, annotated frame and accepted face crop is registered in `capture_images`; face crops reference their original frame where available.
- Business events (`CaptureEvent`, `QualityEvent`, `SaveEvent`, `CaptureManifest`, etc.) are persisted in `business_events`.
- Python application logs are written asynchronously in batches to `application_logs`. The writer reports persistence failures to stderr to avoid recursive logging.
- `GET /api/captures`, `GET /api/events` and `GET /api/logs` support page/page_size pagination. The album retains its offset/limit compatibility.
- Web pages: `/logs` for business events and `/system-logs` for application logs.

### SQLite operational notes

SQLite uses WAL mode, foreign-key enforcement and a 5-second busy timeout. The database lives under the same persistent `data` volume as image files, but in its own `database/` subdirectory. Use an SSD for long-running deployments. Do not delete `data/secrets/camera.key`; back it up securely alongside the database.

### Upgrade from v0.1.3

1. Stop the v0.1.3 container cleanly.
2. Back up the existing `data/`, `config/` and `.env` directories/files.
3. Replace the application files with v0.2.0 while preserving the existing `data/` and `models/` directories.
4. Build and start with `docker compose build` and `docker compose up -d`.
5. On first startup, Alembic creates the database schema, imports configured camera entries from YAML, indexes existing JPEG files and imports `logs/runtime.jsonl`. Verify `/api/status`, `/api/captures`, `/api/events` and `/api/logs`.
6. After the import, manage camera records through the database in future releases; editing YAML no longer updates existing database records.

Do not test the migration against the only copy of production data. The migration creates a new database and does not delete existing image files.


## 9. v0.4.0 Web camera management

- Open `/cameras` to list, add, edit, enable/disable, test or delete cameras.
- Camera ID is immutable after creation. IDs accept letters, digits, underscore, dot and hyphen.
- Main RTSP stream is required. A configured substream enables dual-stream mode (substream detection + timestamp-matched main-stream capture); without a substream, the camera uses single-stream mode (main stream for both).
- Detection FPS is configurable per camera (1–30).
- RTSP URLs remain Fernet-encrypted in SQLite. The API never returns a decrypted URL. The Web form uses password inputs; when editing, leave the main URL blank to preserve it. The substream can be preserved, replaced, or explicitly removed to switch to single-stream mode.
- Editing or enabling/disabling a camera stops/restarts only that camera's RTSP readers. Other camera Workers and the shared inference scheduler remain running. Adding an enabled camera starts its Worker immediately; disabling or deleting stops it. Historical capture files and capture metadata are not deleted when a camera is deleted.
- `PUT /api/cameras/{camera_id}` and enable/disable operations require the current `revision`. Stale writes return HTTP 409 rather than silently overwriting newer configuration.
- `POST /api/cameras/{camera_id}/test` performs a one-shot main-stream connection/read test with a bounded timeout; it does not create a persistent Worker. At most two tests run concurrently. Error responses do not expose RTSP credentials.
- `GET /api/cameras` returns paginated camera metadata and runtime state only; stream URLs and passwords are never returned.
- No schema migration is required for v0.4.0: the v0.2.0 schema already contains the camera revision, stream-mode, detection-FPS and camera-status fields used by this release.

### Camera API examples

```http
GET    /api/cameras?page=1&page_size=50
POST   /api/cameras
PUT    /api/cameras/{camera_id}
PUT    /api/cameras/{camera_id}/enabled
POST   /api/cameras/{camera_id}/test
DELETE /api/cameras/{camera_id}
```

Create body: `{"id":"gate-01","name":"North Gate","main_url":"rtsp://user:password@camera/main","sub_url":"rtsp://user:password@camera/sub","detection_fps":3}`. Store and back up `data/secrets/camera.key` securely; losing it makes stored stream URLs unrecoverable.


## System settings and cloud storage

- Added `/settings` for SQLite-backed MQTT and S3-compatible storage configuration.
- MQTT publishes the six supported business events through a durable SQLite Outbox.
- S3-compatible storage supports AWS S3 and MinIO-style custom endpoints. Access key, secret key and MQTT password are encrypted with the existing `FACE_SECRET_PATH` key; secret values are never returned by the settings API.
- Added bounded background upload queue. The Album page toggle controls automatic upload of newly captured artifacts; each stored image can also be uploaded manually. Upload state and object key are persisted in SQLite.
- Added Alembic migration `0002_cloud_storage` for image upload state. Existing local image files remain untouched.
- Optional dependencies: `boto3` and `paho-mqtt`.

### Cloud setup

Open `/settings`, configure and save MQTT and/or S3, then use **发送测试消息** to verify that the Broker accepts a real MQTT publication. For MinIO, set its internal or reachable S3 endpoint URL, bucket, access key and secret key. In `/album`, enable **自动上传新抓拍至云存储** to upload future captures asynchronously. The toggle does not retroactively upload old images; use each image's **上传云存储** action for historical files.

Back up `data/database/gateway.db` and `data/secrets/camera.key` together. Losing the key makes encrypted camera and cloud credentials unrecoverable.


## 10. v0.4.1 database migration reliability

v0.4.1 fixes an SQLite/Alembic startup defect that could leave application tables created while the `alembic_version` row was not committed. On the next startup Alembic then replayed `0001_initial_schema` and failed with `table cameras already exists`.

- Alembic now runs with an explicit SQLAlchemy transaction so schema changes and the revision marker are committed together.
- Startup migration is protected by a cross-process lock file (`gateway.db.migrate.lock`) so two application instances cannot migrate the same SQLite file simultaneously.
- If `alembic_version` is missing or empty, the application inspects required tables, columns and indexes. It stamps only a schema that matches a known release (`0001_initial_schema` or `0002_cloud_storage`); partial or unknown schemas fail safely without dropping or recreating tables.
- Before recovering an unversioned database or applying a forward migration, the application creates a consistent SQLite online backup under `data/database/backups/`. The backup includes committed WAL data and is checked with `PRAGMA quick_check`.
- After migration, SQLite `quick_check` and `foreign_key_check` are run. A failed migration stops startup; it never deletes the database or automatically recreates tables.
- A database already at the current revision is not backed up again on every restart.

**Upgrade instruction:** stop the previous container, preserve the complete `data/` directory (especially `data/database/` and `data/secrets/`), replace the application files, then run `docker compose build` and `docker compose up -d`. Do not delete `gateway.db`, `alembic_version`, or the `backups/` directory to work around migration errors. If the schema is not recognized, retain the database and logs and resolve it with a schema-specific recovery plan.


## 11. v0.4.2 MQTT event integration

v0.4.2 removes RabbitMQ and replaces it with MQTT for APP notifications and downstream business-service integration. The MQTT Broker is independently deployed (for example, Mosquitto, EMQX, or the Home Assistant MQTT add-on).

### Event contract

All events are published to `{topic_prefix}/{gateway_id}/events` as UTF-8 JSON. The envelope contains `event_id`, `event_type`, `gateway_id`, `timestamp_ms` and `data`. The configured QoS is used (default QoS 1); `retain` is always false for event messages.

| Event | Meaning | Consumer purpose |
|---|---|---|
| `camera.online` | Required stream(s) connected | APP reminder |
| `camera.offline` | A previously online camera lost its required stream(s) | APP reminder |
| `capture.created` | Original capture frame was written locally | APP reminder |
| `capture.saved` | Capture image metadata was indexed locally | APP reminder |
| `upload.complete` | An image was uploaded successfully to S3-compatible storage | Downstream business services, such as face verification |
| `upload.error` | An image upload failed or the automatic-upload queue was full | APP reminder |

`upload.complete.data` includes `capture_id`, `camera_id`, `image_id`, `image_type`, `image_key`, `bucket`, `image_sha256`, `file_size`, `content_type`, `storage_provider` and `uploaded_at_ms`. It is emitted once per successfully uploaded image artifact. A capture may therefore produce multiple upload events (for example, original frame, annotated frame and accepted face crops); consumers should filter `image_type` as appropriate. The gateway does not perform face identity recognition.

`capture.created` is emitted after the original frame JPEG is written. `capture.saved` is emitted after the original frame has a corresponding SQLite image record. Startup while a camera is initially disconnected does not generate an offline alert; online/offline events represent observed connection-state transitions.

### Delivery and recovery

- MQTT credentials are encrypted using `data/secrets/camera.key`.
- MQTT QoS acknowledgement is awaited before an Outbox row is marked `published`.
- When MQTT is enabled, supported events are inserted into `business_events` and `mqtt_outbox` in the same SQLite transaction. Events generated while MQTT is disabled remain local and are not backfilled when MQTT is later enabled.
- When the Broker is unreachable, Outbox rows remain pending and retry with exponential backoff (capped at one hour). They are not silently discarded.
- QoS 1 can redeliver after a crash between Broker acknowledgement and local Outbox completion. Every event therefore has a stable `event_id`; consumers should implement idempotency/deduplication by this ID.
- The **发送测试消息** button publishes a real `gateway.test` message to `{topic_prefix}/{gateway_id}/test`; this diagnostic message is not one of the six business events.
- `/api/system-settings/mqtt-status` reports configuration state, publisher connection state and pending Outbox count.

### Upgrade from v0.4.1

1. Stop the existing container and back up the entire `data/` directory.
2. Replace application files with v0.4.2 while preserving `data/`, `models/`, `config/` and `.env`.
3. Run `docker compose build` and `docker compose up -d`. Alembic applies `0003_mqtt_outbox` and preserves existing camera, capture, event and S3 metadata.
4. Open `/settings`, configure the MQTT Broker, Gateway ID, Client ID and topic prefix, save, then click **发送测试消息**.
5. Subscribe a test client to `edge-face-gateway/{gateway_id}/#` (or your configured prefix) and verify the test event. Then validate the six business events with actual cameras and S3 storage.

The MQTT Broker is not bundled into the gateway Compose file. It can be the existing Home Assistant MQTT Broker or a separately managed Mosquitto/EMQX service. Home Assistant MQTT Discovery and inbound control commands are not included in v0.4.2.
