# Server Deployment

This deployment runs the production frontend and one API worker in separate
containers. It does not change the stock-selection rules or enable a scheduled
scan. It does enable daily close monitoring of saved buy alerts at 16:00 China
time. Docker restarts the containers after a host reboot.

## Layout

- Application: `/home/xisungod/stock_mine`
- Persistent database and market caches: `../data`, relative to this directory
- Private HTTP endpoint on the server: `127.0.0.1:18765`
- Lucky upstream: `http://192.168.0.10:5175`, published on `0.0.0.0:5175`
- API: internal Docker network only, not published on the host
- Timezone: `Asia/Shanghai`
- Close monitoring: `STOCK_MINE_MONITOR_SCHEDULE_ENABLED=1`, daily at 16:00 on
  exchange sessions, with bounded retries and startup catch-up
- Data provider: the same AKShare installation used by the local environment;
  AData is not installed, so `auto` uses AKShare.

The API has no application login. The 5175 listener is intended for Lucky on the
trusted local network. Enable HTTPS and access authentication in Lucky for both
the frontend and every `/api/` endpoint. Do not forward port 5175 directly from
the public Internet; the application does not enforce authentication itself.

## Build And Start

Build the frontend with `npm ci` followed by `npm run build` in `frontend`.
Prepare Linux x86-64 / Python 3.12 wheels on a machine with working PyPI access:

```sh
PYTHON=.venv/bin/python sh deploy/prepare_wheels.sh
```

Transfer source, `deploy/wheels` and `frontend/dist`, excluding `.venv`,
`node_modules`, generated reports, secrets and caches. The image installs from
these wheels without downloading packages on the server. Do not overwrite the
server's live database during subsequent code updates.

From the application directory on the server:

```sh
sudo env DOCKER_BUILDKIT=0 docker compose -f deploy/compose.yaml build
sudo docker compose -f deploy/compose.yaml up -d --wait
sudo docker compose -f deploy/compose.yaml ps
curl --fail http://127.0.0.1:18765/api/health
```

The dependency versions and base image digests are frozen in this directory.
AKShare's Linux dependencies use `py-mini-racer==0.6.0` and `akracer==0.0.14`,
as required by its package metadata, rather than macOS's `mini-racer`.
The existing server has Compose but no Buildx, so the build command explicitly
uses its installed classic builder.
Keep one Uvicorn worker: running scan/backtest state is held in process memory.
The existing four stock-fetch workers are unchanged.

## Access Over SSH

From your own computer:

```sh
ssh -N -L 127.0.0.1:5175:127.0.0.1:18765 xisungod@openclaw.929772.xyz
```

Then open `http://127.0.0.1:5175`. The computation and saved data are on the server,
not the local development backend. Closing the tunnel does not stop the server.
Reconnect the tunnel after a network change or computer restart.

Direct access by domain additionally requires an externally reachable forwarded
port, a valid TLS certificate and an authentication gateway. Existing services
on ports 80 and 3210 must not be replaced.

## Access Through Lucky

- Domain: `stock.929772.xyz`
- Upstream protocol: HTTP
- Upstream address: `192.168.0.10:5175`
- Proxy the whole site, including `/api/`, without stripping or adding a path.
- Keep HTTPS termination and access authentication on Lucky.

Port 5175 on the server is a Docker mapping to the production Nginx gateway,
not a Vite development server. The separate Mac SSH tunnel also uses local
port 5175, but it is not the address that Lucky should connect to.

Apply a gateway-only configuration change without restarting scan jobs:

```sh
sudo docker compose --ansi never -f deploy/compose.yaml config --quiet
sudo docker compose --ansi never -f deploy/compose.yaml up -d --no-deps --wait --wait-timeout 90 web
curl --fail http://192.168.0.10:5175/api/health
```

## Logs And Updates

```sh
sudo docker compose -f deploy/compose.yaml logs --tail=100
sudo docker compose -f deploy/compose.yaml restart api
```

Container logs rotate at 10 MB, keeping three files per service. Container memory
limits are 1792 MB for the API and 128 MB for the frontend on this 4 GB host.

Before updating, back up SQLite with its online backup API or `sqlite3 .backup`.
Do not copy a live database file directly. Preserve the whole data directory and
never run `docker compose down -v` as an update procedure.

The initial backup preserves local scans, historical alerts and the watchlist.
Subsequent local and server changes are independent; there is no automatic
two-way synchronization.

## Alert Monitoring

The `/?view=alert-monitor` view tracks stored real-data buy alerts against the
unadjusted close on their actual selection date. New `alert_monitor_*` tables
store imported entries, daily observations and job progress; the original scan
and watchlist tables are not rewritten.

Check the timer and last job through `GET /api/alert-monitor/status`.
`POST /api/alert-monitor/refresh` starts a manual update and returns immediately.
At most one monitor job runs at a time; fetching is limited to two threads.
Keep the single API worker and persistent data volume when updating the image.
The server does not depend on a Codex task or an open browser for its timer.
