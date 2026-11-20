from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator
from urllib.parse import quote, urlsplit

import psutil
import cv2
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field
import re
import threading
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse

from .engine import RuntimeEngine
from sqlalchemy import select, desc
from core.infrastructure.database.models import (
    ApplicationLog,
    BusinessEvent,
    Camera,
    CameraStatus,
    CaptureImage,
    MqttOutboxEvent,
)
from core.infrastructure.database.pagination import Page, paginate
from .config import CameraConfig
from core.infrastructure.auth import (
    AuthenticationError,
    AuthService,
    DuplicateUsernameError,
    LastActiveUserError,
    PASSWORD_MIN_LENGTH,
    SESSION_COOKIE,
    SESSION_TTL_MS,
    UserNotFoundError,
)

log = logging.getLogger(__name__)


SHELL = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Edge Face Gateway</title><style>
:root{color-scheme:dark;--bg:#0c1119;--panel:#151e2b;--line:#29384a;--muted:#9aabc0;--accent:#83bfff;--green:#6bd6a0}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#e8eef7;font:14px system-ui,Segoe UI,sans-serif}header{padding:18px 24px;border-bottom:1px solid var(--line);display:flex;gap:24px;align-items:center;flex-wrap:wrap}h1{font-size:20px;margin:0}nav{display:flex;gap:8px}nav a{color:var(--muted);text-decoration:none;padding:8px 12px;border-radius:8px}nav a.active,nav a:hover{background:#223247;color:white}.wrap{padding:22px;max-width:1500px;margin:auto}.stats{color:var(--muted);margin:0 0 18px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;min-width:0}.feed{width:100%;background:#05080d;min-height:180px;object-fit:contain;border-radius:8px}.meta{font-size:13px;color:var(--muted);line-height:1.8}.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:18px}select,input,button{background:#172333;color:#e8eef7;border:1px solid var(--line);padding:9px 11px;border-radius:8px}button{cursor:pointer}button:hover{border-color:var(--accent)}.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px}.gallery img{width:100%;aspect-ratio:16/10;object-fit:contain;background:#080c12;border-radius:8px}.gallery .face{aspect-ratio:1/1;object-fit:contain}.pill{display:inline-block;border:1px solid var(--line);padding:3px 7px;border-radius:20px;color:var(--muted);font-size:11px}.event{display:grid;grid-template-columns:160px 130px 1fr;gap:12px;padding:12px 8px;border-bottom:1px solid var(--line);align-items:start}.event small{color:var(--muted)}.empty{color:var(--muted);padding:28px;text-align:center}.section-title{font-size:17px;margin:0 0 14px}.muted{color:var(--muted)}a{color:var(--accent)}@media(max-width:650px){.event{grid-template-columns:1fr;gap:4px}.wrap{padding:14px}header{padding:14px}}
</style></head><body><header><h1>Edge Face Gateway <span class="pill">v0.4.2 · CPU</span></h1><nav><a href="/" data-page="home">实时监控</a><a href="/cameras" data-page="cameras">摄像头管理</a><a href="/settings" data-page="settings">系统设置</a><a href="/album" data-page="album">相册</a><a href="/logs" data-page="logs">业务事件</a><a href="/system-logs" data-page="system-logs">运行日志</a></nav></header><main class="wrap">__BODY__</main><script>document.querySelectorAll('nav a').forEach(a=>{if(a.dataset.page==='__PAGE__')a.classList.add('active')});</script>__SCRIPT__</body></html>"""

HOME_BODY = r'''<p id="system" class="stats">正在读取系统指标…</p><div id="grid" class="grid"></div>'''
HOME_SCRIPT = r'''<script>
async function refresh(){try{const d=await (await fetch('/api/status')).json();const s=d.system;document.getElementById('system').textContent=`进程 CPU ${s.process_cpu_percent}%（${s.process_cpu_cores} 核） · 整机 CPU ${s.host_cpu_percent}% · RSS ${s.process_rss_mb} MB · 运行 ${d.uptime_seconds}s`;const g=document.getElementById('grid');for(const c of d.cameras){let id='cam-'+c.camera_id,e=document.getElementById(id);if(!e){e=document.createElement('section');e.className='card';e.id=id;e.innerHTML='<h3></h3><img class="feed"><div class="meta"></div>';g.appendChild(e);e.querySelector('img').src='/stream/'+encodeURIComponent(c.camera_id)}e.querySelector('h3').textContent=c.camera_id+' · '+c.mode.toUpperCase();e.querySelector('.meta').textContent=`main ${c.main_connected?'connected':'disconnected'} · sub ${c.sub_connected===null?'—':(c.sub_connected?'connected':'disconnected')} | FPS ${c.inference_fps} | inference ${c.last_inference_ms} ms (P95 ${c.p95_inference_ms}) | faces ${c.last_face_count} | captures ${c.captures_saved} | misses ${c.capture_misses} | reconnects ${c.main_reconnects}/${c.sub_reconnects??0}`+(c.last_error?' | ERROR: '+c.last_error:'')}}catch(e){document.getElementById('system').textContent='指标读取失败：'+e}}refresh();setInterval(refresh,2000);
</script>'''

ALBUM_BODY = r'''<div class="toolbar"><label><input type="checkbox" id="auto-upload"> 自动上传新抓拍至云存储</label><span class="muted" id="upload-setting-status"></span><label>摄像头 <select id="camera"><option value="">全部摄像头</option></select></label><label>图片类型 <select id="kind"><option value="all">全部</option><option value="frame">原始帧</option><option value="annotated">评分框</option><option value="face">人脸裁剪</option></select></label><button id="reload">刷新</button><span class="muted" id="count"></span></div><div id="gallery" class="gallery"></div><div class="toolbar"><button id="more">加载更多</button></div>'''
ALBUM_SCRIPT = r"""<script>
const cameraFilter=document.getElementById('camera'),kindFilter=document.getElementById('kind'),gallery=document.getElementById('gallery'),moreButton=document.getElementById('more'),countLabel=document.getElementById('count'),autoUpload=document.getElementById('auto-upload');
async function loadUploadSetting(){const d=await(await fetch('/api/system-settings')).json();cloudEnabled=!!d.s3?.enabled;autoUpload.checked=d.auto_upload;autoUpload.disabled=!cloudEnabled;document.getElementById('upload-setting-status').textContent=!cloudEnabled?'请先在系统设置配置 S3 服务':(d.auto_upload?'新抓拍将自动上传':'仅保存在本地')}
autoUpload.onchange=async()=>{try{const r=await fetch('/api/system-settings/auto-upload',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:autoUpload.checked})});if(!r.ok){const d=await r.json().catch(()=>({}));throw Error(d.detail||'保存失败')}loadUploadSetting()}catch(e){autoUpload.checked=!autoUpload.checked;alert(e.message)}};
let offset=0,limit=48,cloudEnabled=false;
async function loadAlbum(reset=false){if(reset){offset=0;gallery.innerHTML=''}const q=new URLSearchParams({offset,limit});if(cameraFilter.value)q.set('camera_id',cameraFilter.value);if(kindFilter.value!=='all')q.set('kind',kindFilter.value);const d=await(await fetch('/api/captures?'+q)).json();if(reset){cameraFilter.innerHTML='<option value="">全部摄像头</option>';for(const c of d.cameras)cameraFilter.add(new Option(c,c))}countLabel.textContent=`共 ${d.total} 项 · 当前显示 ${gallery.children.length+d.items.length} 项`;for(const x of d.items){const el=document.createElement('article');el.className='card';const img=document.createElement('img');img.src=x.thumbnail_url;img.loading='lazy';img.className=x.kind==='face'?'face':'';img.alt=x.kind;const title=document.createElement('div');const name=document.createElement('b');name.textContent=x.camera_id;const badge=document.createElement('span');badge.className='pill';badge.textContent=x.kind_label;title.append(name,document.createTextNode(' '),badge);const meta=document.createElement('div');meta.className='meta';meta.textContent=`${x.timestamp} · ${x.width}×${x.height}${x.score!==null?' · score '+x.score:''}`;const actions=document.createElement('div');actions.className='toolbar';const link=document.createElement('a');link.href=x.url;link.target='_blank';link.rel='noopener';link.textContent='打开原图';actions.append(link);if(x.id && cloudEnabled){const cloud=document.createElement('span');cloud.className='pill';cloud.textContent=x.cloud_upload_status==='uploaded'?'已上传':(x.cloud_upload_status==='failed'?'上传失败':'本地');actions.append(cloud);const upload=document.createElement('button');upload.textContent=x.cloud_upload_status==='uploaded'?'重新上传':'上传云存储';upload.onclick=async()=>{upload.disabled=true;upload.textContent='上传中…';try{const res=await fetch('/api/captures/'+x.id+'/upload',{method:'POST'});const out=await res.json();if(!res.ok)throw Error(out.detail||'上传失败');cloud.textContent='已上传';upload.textContent='重新上传'}catch(e){upload.textContent='上传失败';alert(e.message)}finally{upload.disabled=false}};actions.append(upload)}el.append(img,title,meta,actions);gallery.append(el)}offset+=d.items.length;moreButton.disabled=offset>=d.total;if(!d.items.length&&offset===0)gallery.innerHTML='<div class="empty">暂无抓拍图片</div>'}
document.getElementById('reload').onclick=()=>loadAlbum(true);cameraFilter.onchange=()=>loadAlbum(true);kindFilter.onchange=()=>loadAlbum(true);moreButton.onclick=()=>loadAlbum();loadAlbum(true);loadUploadSetting();
</script>"""

SETTINGS_BODY = r'''<div class="grid"><section class="card"><h2 class="section-title">MQTT 消息服务</h2><form id="mqtt-form"><p><label><input type="checkbox" id="mqtt-enabled"> 启用 MQTT 事件发布</label></p><p>Broker 主机<br><input id="mqtt-host" required style="width:100%" placeholder="例如 mqtt.local"></p><p>端口<br><input id="mqtt-port" type="number" min="1" max="65535" value="1883" required></p><p>用户名<br><input id="mqtt-user" style="width:100%"></p><p>密码<br><input id="mqtt-password" type="password" autocomplete="new-password" placeholder="留空表示保持原密码" style="width:100%"></p><p>Client ID<br><input id="mqtt-client-id" required style="width:100%"></p><p>Gateway ID<br><input id="mqtt-gateway-id" required style="width:100%"></p><p>Topic 前缀<br><input id="mqtt-topic-prefix" required style="width:100%"></p><p>QoS<br><select id="mqtt-qos"><option value="0">0 - At most once</option><option value="1" selected>1 - At least once（推荐）</option><option value="2">2 - Exactly once</option></select></p><p>Keep Alive（秒）<br><input id="mqtt-keep-alive" type="number" min="10" max="3600" value="60"></p><p><label><input type="checkbox" id="mqtt-tls"> 使用 TLS</label></p><div class="toolbar"><button type="submit">保存 MQTT 配置</button><button type="button" id="mqtt-test">发送测试消息</button></div><div class="meta" id="mqtt-result"></div><div class="meta" id="mqtt-status"></div></form></section><section class="card"><h2 class="section-title">S3 兼容对象存储</h2><form id="s3-form"><p><label><input type="checkbox" id="s3-enabled"> 启用 S3 云存储</label></p><p>Endpoint URL（MinIO 可填写内网地址）<br><input id="s3-endpoint" placeholder="https://s3.example.com" style="width:100%"></p><p>Region<br><input id="s3-region" value="us-east-1" style="width:100%"></p><p>Bucket<br><input id="s3-bucket" style="width:100%"></p><p>Access Key<br><input id="s3-access" style="width:100%"></p><p>Secret Key<br><input id="s3-secret" type="password" autocomplete="new-password" placeholder="留空表示保持原密钥" style="width:100%"></p><p>对象前缀<br><input id="s3-prefix" value="edge-face" style="width:100%"></p><p><label><input type="checkbox" id="s3-ssl" checked> 使用 HTTPS</label></p><div class="toolbar"><button type="submit">保存 S3 配置</button><button type="button" id="s3-test">测试 Bucket</button></div><div class="meta" id="s3-result"></div></form></section></div><section class="card"><h2 class="section-title">MQTT 事件清单</h2><div class="meta">camera.online、camera.offline、capture.created、capture.saved、upload.error 用于 APP 提醒；upload.complete 用于触发人脸验证等后续业务。Topic：{topic_prefix}/{gateway_id}/events。事件采用 QoS 配置、retain=false，并通过 SQLite Outbox 在 Broker 恢复后重试。</div></section><p class="muted">MQTT、S3 凭据在数据库中加密保存，API 不返回明文。加密密钥位于 data/secrets/camera.key，请与数据库一起纳入备份计划。</p>'''
SETTINGS_SCRIPT = r'''<script>
async function api(url,opt={}){const r=await fetch(url,{headers:{'Content-Type':'application/json'},...opt});const d=await r.json().catch(()=>({}));if(!r.ok)throw Error(d.detail||'请求失败');return d}
function val(id){return document.getElementById(id).value}function checked(id){return document.getElementById(id).checked}
async function load(){const d=await api('/api/system-settings'),m=d.mqtt,s=d.s3;document.getElementById('mqtt-enabled').checked=m.enabled;document.getElementById('mqtt-host').value=m.host;document.getElementById('mqtt-port').value=m.port;document.getElementById('mqtt-user').value=m.username||'';document.getElementById('mqtt-client-id').value=m.client_id;document.getElementById('mqtt-gateway-id').value=m.gateway_id;document.getElementById('mqtt-topic-prefix').value=m.topic_prefix;document.getElementById('mqtt-qos').value=m.qos;document.getElementById('mqtt-keep-alive').value=m.keep_alive;document.getElementById('mqtt-tls').checked=m.use_tls;document.getElementById('mqtt-password').placeholder=m.has_password?'已配置（留空保持原密码）':'可选：请输入密码';document.getElementById('s3-enabled').checked=s.enabled;document.getElementById('s3-endpoint').value=s.endpoint_url||'';document.getElementById('s3-region').value=s.region;document.getElementById('s3-bucket').value=s.bucket;document.getElementById('s3-access').value='';document.getElementById('s3-access').placeholder=s.has_access_key?'已配置（留空保持原 Access Key）':'请输入 Access Key';document.getElementById('s3-secret').placeholder=s.has_secret_key?'已配置（留空保持原密钥）':'请输入 Secret Key';document.getElementById('s3-prefix').value=s.prefix;document.getElementById('s3-ssl').checked=s.use_ssl;refreshMqttStatus()}
document.getElementById('mqtt-form').onsubmit=async e=>{e.preventDefault();try{await api('/api/system-settings',{method:'PUT',body:JSON.stringify({mqtt:{enabled:checked('mqtt-enabled'),host:val('mqtt-host'),port:Number(val('mqtt-port')),username:val('mqtt-user'),password:val('mqtt-password'),client_id:val('mqtt-client-id'),gateway_id:val('mqtt-gateway-id'),topic_prefix:val('mqtt-topic-prefix'),qos:Number(val('mqtt-qos')),keep_alive:Number(val('mqtt-keep-alive')),use_tls:checked('mqtt-tls')}})});document.getElementById('mqtt-password').value='';document.getElementById('mqtt-result').textContent='MQTT 配置已保存';refreshMqttStatus()}catch(x){document.getElementById('mqtt-result').textContent=x.message}};
document.getElementById('s3-form').onsubmit=async e=>{e.preventDefault();try{await api('/api/system-settings',{method:'PUT',body:JSON.stringify({s3:{enabled:checked('s3-enabled'),endpoint_url:val('s3-endpoint'),region:val('s3-region'),bucket:val('s3-bucket'),access_key:val('s3-access'),secret_key:val('s3-secret'),prefix:val('s3-prefix'),use_ssl:checked('s3-ssl')}})});document.getElementById('s3-secret').value='';document.getElementById('s3-result').textContent='配置已保存'}catch(x){document.getElementById('s3-result').textContent=x.message}};
document.getElementById('mqtt-test').onclick=async()=>{const o=document.getElementById('mqtt-result');o.textContent='正在连接 Broker 并发布测试消息…';try{const d=await api('/api/system-settings/test-mqtt',{method:'POST'});o.textContent=d.message;refreshMqttStatus()}catch(e){o.textContent=e.message}};document.getElementById('s3-test').onclick=async()=>{const o=document.getElementById('s3-result');o.textContent='测试中…';try{const d=await api('/api/system-settings/test-s3',{method:'POST'});o.textContent=d.message}catch(e){o.textContent=e.message}};async function refreshMqttStatus(){try{const s=await api('/api/system-settings/mqtt-status');document.getElementById('mqtt-status').textContent=`状态：${s.enabled?(s.connected?'已连接':'等待连接'):'未启用'} · Outbox 待发送 ${s.pending_events} 条 · Topic ${s.topic}`}catch(e){}}load();setInterval(refreshMqttStatus,5000);
</script>'''

LOGS_BODY = r'''<div class="toolbar"><label>事件类型 <select id="type"><option value="">全部事件</option><option>camera.online</option><option>camera.offline</option><option>capture.created</option><option>capture.saved</option><option>capture.missed</option><option>capture.failed</option><option>capture.detected</option><option>capture.score_rejected</option><option>quality.face_rejected</option><option>quality.face_accepted</option><option>artifact.save_failed</option><option>artifact.saved</option><option>upload.complete</option><option>upload.error</option></select></label><label>摄像头 <select id="camera"><option value="">全部摄像头</option></select></label><button id="reload">刷新</button><span id="count" class="muted"></span></div><section class="card"><div id="events"></div></section><div class="toolbar"><button id="prev">上一页</button><button id="next">下一页</button></div>'''
LOGS_SCRIPT = r"""<script>
const typeFilter=document.getElementById('type'),cameraFilter=document.getElementById('camera'),eventList=document.getElementById('events'),countLabel=document.getElementById('count');let page=1,totalPages=0;
function formatEventTime(value){if(value===null||value===undefined||value==='')return '—';const date=new Date(value);return Number.isNaN(date.getTime())?'—':date.toLocaleString()}
async function loadEvents(reset=false){if(reset)page=1;const q=new URLSearchParams({page,page_size:50});if(typeFilter.value)q.set('event_type',typeFilter.value);if(cameraFilter.value)q.set('camera_id',cameraFilter.value);const d=await(await fetch('/api/events?'+q)).json();const selected=cameraFilter.value;cameraFilter.innerHTML='<option value="">全部摄像头</option>';for(const c of d.cameras)cameraFilter.add(new Option(c,c));cameraFilter.value=selected;totalPages=d.pagination.total_pages;countLabel.textContent=`共 ${d.total} 条 · 第 ${page}/${totalPages||0} 页`;eventList.innerHTML='';for(const e of d.items){const row=document.createElement('div');row.className='event';const times=document.createElement('div');times.className='meta';const created=document.createElement('small');created.textContent='创建时间：'+formatEventTime(e.created_at_ms);const pushed=document.createElement('small');pushed.textContent='推送时间：'+(e.published_at_ms?formatEventTime(e.published_at_ms):(e.push_status==='pending'?'待推送':(e.push_status==='not_queued'?'未入队':'—')));times.append(created,document.createElement('br'),pushed);const typ=document.createElement('div');const name=document.createElement('b');name.textContent=e.event_type||'—';const status=document.createElement('span');status.className='pill';status.textContent=e.status||'—';typ.append(name,document.createElement('br'),status);const detail=document.createElement('div');detail.className='meta';detail.textContent=[e.camera_id,e.artifact,e.path,e.face_count!==undefined?'faces '+e.face_count:'',e.min_capture_score!==undefined?'min score '+e.min_capture_score:'',e.rejected_faces?e.rejected_faces.map(f=>'face '+f.face_index+': '+f.confidence).join(', '):'',e.accepted_face_count!==undefined?'accepted '+e.accepted_face_count:'',e.rejected_face_count!==undefined?'rejected '+e.rejected_face_count:'',e.reasons?e.reasons.join(', '):'',e.error].filter(Boolean).join(' · ');row.append(times,typ,detail);eventList.append(row)}if(!d.items.length)eventList.innerHTML='<div class="empty">暂无该类型事件。</div>';document.getElementById('prev').disabled=page<=1;document.getElementById('next').disabled=page>=totalPages}
document.getElementById('reload').onclick=()=>loadEvents(true);typeFilter.onchange=()=>loadEvents(true);cameraFilter.onchange=()=>loadEvents(true);document.getElementById('prev').onclick=()=>{if(page>1){page--;loadEvents()}};document.getElementById('next').onclick=()=>{if(page<totalPages){page++;loadEvents()}};loadEvents();
</script>"""

SYSTEM_LOGS_BODY = r'''<div class="toolbar"><label>级别 <select id="level"><option value="">全部级别</option><option>DEBUG</option><option>INFO</option><option>WARNING</option><option>ERROR</option><option>CRITICAL</option></select></label><label>分类 <select id="category"><option value="">全部分类</option><option>SYSTEM</option><option>CAMERA</option><option>VIDEO</option><option>DETECTION</option><option>QUALITY</option><option>CAPTURE</option><option>STORAGE</option><option>DATABASE</option><option>API</option></select></label><button id="reload">刷新</button><span id="count" class="muted"></span></div><section class="card"><div id="logs"></div></section><div class="toolbar"><button id="prev">上一页</button><button id="next">下一页</button></div>'''
SYSTEM_LOGS_SCRIPT = r"""<script>
let page=1,totalPages=0;const list=document.getElementById('logs');async function load(reset=false){if(reset)page=1;const q=new URLSearchParams({page,page_size:50});const level=document.getElementById('level').value,category=document.getElementById('category').value;if(level)q.set('level',level);if(category)q.set('category',category);const d=await(await fetch('/api/logs?'+q)).json();totalPages=d.pagination.total_pages;document.getElementById('count').textContent=`共 ${d.pagination.total} 条 · 第 ${page}/${totalPages||0} 页`;list.innerHTML='';for(const x of d.items){const row=document.createElement('div');row.className='event';const time=document.createElement('small');time.textContent=new Date(x.timestamp_ms).toLocaleString();const type=document.createElement('div');const level=document.createElement('b');level.textContent=x.level;const category=document.createElement('span');category.className='pill';category.textContent=x.category;type.append(level,document.createElement('br'),category);const message=document.createElement('div');message.className='meta';message.textContent=[x.logger,x.camera_id,x.message,x.exception].filter(Boolean).join(' · ');row.append(time,type,message);list.append(row)}if(!d.items.length)list.innerHTML='<div class="empty">暂无运行日志</div>';document.getElementById('prev').disabled=page<=1;document.getElementById('next').disabled=page>=totalPages}
document.getElementById('reload').onclick=()=>load(true);document.getElementById('level').onchange=()=>load(true);document.getElementById('category').onchange=()=>load(true);document.getElementById('prev').onclick=()=>{if(page>1){page--;load()}};document.getElementById('next').onclick=()=>{if(page<totalPages){page++;load()}};load();
</script>"""

LOGIN_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>系统登录 · Edge Face Gateway</title>
<style>
*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;background:#0c1119;color:#e8eef7;font:14px system-ui,Segoe UI,sans-serif}.login-card{width:min(440px,calc(100% - 32px));padding:28px;background:#151e2b;border:1px solid #29384a;border-radius:12px}h1{font-size:20px;margin:0 0 22px}label{display:block;margin:16px 0 6px}input{width:100%;margin-top:6px;padding:10px 12px;background:#172333;color:#e8eef7;border:1px solid #29384a;border-radius:8px}button{width:100%;margin-top:18px;padding:10px;background:#223247;color:#fff;border:1px solid #29384a;border-radius:8px;cursor:pointer}.error{min-height:1.5em;margin-top:12px;color:#ff9b9b}
</style>
</head>
<body>
<main>__BODY__</main>
__SCRIPT__
</body>
</html>"""
LOGIN_BODY = r'''<section class="login-card"><h1>Edge Face Gateway</h1><h2>系统登录</h2><form id="login-form"><label>用户名<input id="username" autocomplete="username" required maxlength="128"></label><label>密码<input id="password" type="password" autocomplete="current-password" required></label><button type="submit">登录</button><div id="login-error" class="error"></div></form></section>'''
LOGIN_SCRIPT = r"""<script>
document.getElementById('login-form').onsubmit=async event=>{event.preventDefault();const error=document.getElementById('login-error');error.textContent='';try{const response=await fetch('/api/auth/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:document.getElementById('username').value,password:document.getElementById('password').value})});const data=await response.json();if(!response.ok)throw Error(data.detail||'登录失败');location.replace(data.must_change_password?'/change-password':'/')}catch(e){error.textContent=e.message}};
</script>"""
CHANGE_PASSWORD_BODY = r'''<section class="card" style="max-width:520px;margin:40px auto"><h2 class="section-title">修改初始密码</h2><p class="meta">首次登录必须修改初始密码。新密码至少 6 个字符。</p><form id="password-form"><p><label>当前密码<br><input id="current-password" type="password" autocomplete="current-password" required style="width:100%"></label></p><p><label>新密码<br><input id="new-password" type="password" autocomplete="new-password" minlength="6" maxlength="256" required style="width:100%"></label></p><p><label>确认新密码<br><input id="confirm-password" type="password" autocomplete="new-password" minlength="6" maxlength="256" required style="width:100%"></label></p><button type="submit">保存新密码</button><div id="password-error" class="meta"></div></form></section>'''
CHANGE_PASSWORD_SCRIPT = r"""<script>
document.getElementById('password-form').onsubmit=async event=>{event.preventDefault();const error=document.getElementById('password-error'),password=document.getElementById('new-password').value;if(password!==document.getElementById('confirm-password').value){error.textContent='两次输入的新密码不一致';return}try{const response=await fetch('/api/auth/change-password',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({current_password:document.getElementById('current-password').value,new_password:password})});const data=await response.json();if(!response.ok)throw Error(data.detail||'修改密码失败');location.replace('/')}catch(e){error.textContent=e.message}};
</script>"""
USERS_BODY = r'''<section class="card"><h2 class="section-title">添加系统用户</h2><form id="create-user"><div class="toolbar"><label>用户名 <input id="new-username" required maxlength="128" pattern="[A-Za-z0-9_.-]+"></label><label>初始密码 <input id="new-password" type="password" autocomplete="new-password" minlength="6" maxlength="256" required></label><button type="submit">添加</button></div><div class="meta">所有系统用户均拥有管理员权限；新用户首次登录时必须修改初始密码。</div></form></section><h2 class="section-title">系统用户</h2><div id="users" class="grid"></div><div id="users-message" class="meta"></div><dialog id="user-editor" style="background:#151e2b;color:#e8eef7;border:1px solid #29384a;border-radius:12px;max-width:520px;width:95%;padding:22px"><form id="user-form"><h2>编辑系统用户</h2><label>用户名<br><input id="edit-username" required maxlength="128" pattern="[A-Za-z0-9_.-]+" style="width:100%"></label><p><label><input id="edit-active" type="checkbox"> 启用账号</label></p><label>重置密码（留空不修改）<br><input id="edit-password" type="password" autocomplete="new-password" minlength="6" maxlength="256" style="width:100%"></label><p class="meta">重置密码后，该用户下次登录必须修改密码。</p><div class="toolbar"><button type="submit">保存</button><button type="button" id="cancel-edit">取消</button></div></form></dialog>'''
USERS_SCRIPT = r"""<script>
const usersList=document.getElementById('users'),usersMessage=document.getElementById('users-message'),userEditor=document.getElementById('user-editor'),userForm=document.getElementById('user-form');let editingUser=null;
async function usersApi(url,options={}){const response=await fetch(url,{headers:{'Content-Type':'application/json'},...options}),data=await response.json().catch(()=>({}));if(!response.ok)throw Error(data.detail||'请求失败');return data}
function setMessage(message){usersMessage.textContent=message||''}
async function loadUsers(){try{const data=await usersApi('/api/users');usersList.innerHTML='';for(const user of data.items){const card=document.createElement('section');card.className='card';const title=document.createElement('h3');title.textContent=user.username;const state=document.createElement('div');state.className='meta';state.textContent=(user.is_active?'已启用':'已停用')+(user.must_change_password?' · 下次登录需修改密码':'')+' · 创建于 '+new Date(user.created_at_ms).toLocaleString();const actions=document.createElement('div');actions.className='toolbar';const edit=document.createElement('button');edit.textContent='编辑/重置密码';edit.onclick=()=>editUser(user);const remove=document.createElement('button');remove.textContent='删除';remove.onclick=()=>deleteUser(user);actions.append(edit,remove);card.append(title,state,actions);usersList.append(card)}}catch(e){setMessage(e.message)}}
document.getElementById('create-user').onsubmit=async event=>{event.preventDefault();try{await usersApi('/api/users',{method:'POST',body:JSON.stringify({username:document.getElementById('new-username').value.trim(),password:document.getElementById('new-password').value})});event.target.reset();setMessage('用户已创建');loadUsers()}catch(e){setMessage(e.message)}};
function editUser(user){editingUser=user;document.getElementById('edit-username').value=user.username;document.getElementById('edit-active').checked=user.is_active;document.getElementById('edit-password').value='';userEditor.showModal()}
userForm.onsubmit=async event=>{event.preventDefault();try{await usersApi('/api/users/'+editingUser.id,{method:'PUT',body:JSON.stringify({username:document.getElementById('edit-username').value.trim(),is_active:document.getElementById('edit-active').checked,password:document.getElementById('edit-password').value||null})});userEditor.close();setMessage('用户已更新');loadUsers()}catch(e){setMessage(e.message)}};
async function deleteUser(user){if(!confirm('确定删除用户 '+user.username+'？'))return;try{await usersApi('/api/users/'+user.id,{method:'DELETE'});setMessage('用户已删除');loadUsers()}catch(e){setMessage(e.message)}}
document.getElementById('cancel-edit').onclick=()=>userEditor.close();loadUsers();
</script>"""


def _page(body: str, script: str, page: str) -> str:
    html = SHELL.replace("__BODY__", body).replace("__PAGE__", page)
    controls = '<a href="/users" data-page="users">系统用户</a><button id="logout" type="button">退出登录</button>'
    html = html.replace("</nav>", controls + "</nav>")
    script += r"""<script>document.getElementById('logout').onclick=async()=>{try{await fetch('/api/auth/logout',{method:'POST'})}finally{location.replace('/login')}};</script>"""
    return html.replace("__SCRIPT__", script)


class SystemMetricsSampler:
    """Collect CPU metrics on a fixed cadence, independent of dashboard clients."""

    def __init__(self, process: psutil.Process, logical_cpus: int, interval: float = 1.0):
        import threading
        self.process = process
        self.logical_cpus = logical_cpus
        self.interval = interval
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.process_cpu_raw = 0.0
        self.host_cpu = 0.0
        self.process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        self.thread = threading.Thread(target=self._run, name="system-metrics", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            process_cpu = self.process.cpu_percent(interval=None)
            host_cpu = psutil.cpu_percent(interval=None)
            with self.lock:
                self.process_cpu_raw = process_cpu
                self.host_cpu = host_cpu

    def snapshot(self) -> tuple[float, float]:
        with self.lock:
            return self.process_cpu_raw, self.host_cpu

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)


CAMERAS_BODY = r'''<div class="toolbar"><button id="add">添加摄像头</button><button id="reload">刷新</button><span class="muted" id="message"></span></div><div id="camera-list" class="grid"></div><dialog id="editor" style="background:#151e2b;color:#e8eef7;border:1px solid #29384a;border-radius:12px;max-width:620px;width:95%;padding:22px"><form id="camera-form"><h2 id="form-title">添加摄像头</h2><input type="hidden" id="revision"><label>摄像头 ID<br><input id="camera-id" required maxlength="128" pattern="[A-Za-z0-9_.-]+" style="width:100%"></label><p><label>名称<br><input id="name" required maxlength="200" style="width:100%"></label></p><p><label>主码流 RTSP URL<br><input id="main-url" type="password" autocomplete="new-password" placeholder="rtsp://...（编辑时留空表示不修改）" style="width:100%"></label></p><p><label>子码流 RTSP URL（可选）<br><input id="sub-url" type="password" autocomplete="new-password" placeholder="rtsp://..." style="width:100%"></label></p><p id="single-mode-row" style="display:none"><label><input id="single-mode" type="checkbox"> 切换为单流模式（停止使用子码流）</label></p><p><label>检测帧率 FPS <input id="fps" type="number" min="1" max="30" value="3"></label></p><p><label>最低抓拍分数 <input id="min-capture-score" type="number" min="0" max="1" step="0.01" value="0.78" required></label></p><p><label><input id="enabled" type="checkbox" checked> 启用摄像头</label></p><div class="toolbar"><button type="submit">保存</button><button type="button" id="cancel">取消</button></div><small class="muted">出于安全考虑，已保存的 RTSP 地址不会回显。编辑时主码流地址留空表示保持不变；需要切换为单流模式时勾选对应选项。</small></form></dialog>'''
CAMERAS_SCRIPT = r'''<script>
const list=document.getElementById('camera-list'),dlg=document.getElementById('editor'),form=document.getElementById('camera-form'),msg=document.getElementById('message');let current=null;
async function api(url,opt={}){const r=await fetch(url,{headers:{'Content-Type':'application/json'},...opt});const d=await r.json().catch(()=>({}));if(!r.ok)throw Error(d.detail||'请求失败');return d}
function esc(v){return String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
async function load(){try{const d=await api('/api/cameras');list.innerHTML='';for(const c of d.items){const el=document.createElement('section');el.className='card';el.innerHTML=`<h3>${esc(c.name)} <span class="pill">${esc(c.id)}</span></h3><div class="meta">${c.enabled?'已启用':'已禁用'} · ${c.stream_mode==='dual'?'双流':'单流'} · ${c.detection_fps} FPS · 最低抓拍分数 ${c.min_capture_score} · ${esc(c.runtime_status||'STOPPED')} · revision ${c.revision}</div><div class="toolbar"><button data-action="edit">编辑</button><button data-action="toggle">${c.enabled?'禁用':'启用'}</button><button data-action="test">测试主码流</button><button data-action="preview">预览</button><button data-action="delete">删除</button></div><div class="meta" data-result></div><img class="feed" style="display:none" data-preview>`;el.querySelector('[data-action=edit]').onclick=()=>edit(c);el.querySelector('[data-action=toggle]').onclick=()=>toggle(c);el.querySelector('[data-action=test]').onclick=()=>test(c,el);el.querySelector('[data-action=preview]').onclick=()=>preview(c,el);el.querySelector('[data-action=delete]').onclick=()=>removeCamera(c);list.append(el)}}catch(e){msg.textContent=e.message}}
function edit(c){current=c;document.getElementById('form-title').textContent='编辑摄像头';document.getElementById('camera-id').value=c.id;document.getElementById('camera-id').disabled=true;document.getElementById('name').value=c.name;document.getElementById('revision').value=c.revision;document.getElementById('main-url').value='';document.getElementById('sub-url').value='';document.getElementById('main-url').placeholder=c.has_main_url?'已配置（留空不修改）':'请输入主码流 RTSP URL';document.getElementById('sub-url').placeholder=c.stream_mode==='dual'?'已配置（留空保持原配置）':'请输入子码流 URL（可选）';document.getElementById('single-mode-row').style.display='block';document.getElementById('single-mode').checked=false;document.getElementById('fps').value=c.detection_fps;document.getElementById('min-capture-score').value=c.min_capture_score;document.getElementById('enabled').checked=c.enabled;dlg.showModal()}
document.getElementById('add').onclick=()=>{current=null;form.reset();document.getElementById('form-title').textContent='添加摄像头';document.getElementById('camera-id').disabled=false;document.getElementById('fps').value=3;document.getElementById('min-capture-score').value=0.78;document.getElementById('main-url').placeholder='rtsp://...';document.getElementById('sub-url').placeholder='可选';document.getElementById('single-mode-row').style.display='none';dlg.showModal()};document.getElementById('cancel').onclick=()=>dlg.close();document.getElementById('reload').onclick=load;
form.onsubmit=async e=>{e.preventDefault();const id=document.getElementById('camera-id').value.trim(),main=document.getElementById('main-url').value.trim(),sub=document.getElementById('sub-url').value.trim(),body={name:document.getElementById('name').value.trim(),main_url:main||null,sub_url:sub||null,detection_fps:Number(document.getElementById('fps').value),min_capture_score:Number(document.getElementById('min-capture-score').value),enabled:document.getElementById('enabled').checked};try{if(current){body.revision=Number(document.getElementById('revision').value);body.keep_substream=!document.getElementById('single-mode').checked&&!sub; if(document.getElementById('single-mode').checked)body.sub_url=null;await api('/api/cameras/'+encodeURIComponent(id),{method:'PUT',body:JSON.stringify(body)})}else{body.main_url=main;await api('/api/cameras',{method:'POST',body:JSON.stringify({id,...body})})}dlg.close();msg.textContent='保存成功';load()}catch(e){msg.textContent=e.message}};
async function toggle(c){try{await api('/api/cameras/'+encodeURIComponent(c.id)+'/enabled',{method:'PUT',body:JSON.stringify({enabled:!c.enabled,revision:c.revision})});load()}catch(e){msg.textContent=e.message}}async function removeCamera(c){if(!confirm('确定删除摄像头 '+c.id+'？历史抓拍图片不会删除。'))return;try{await api('/api/cameras/'+encodeURIComponent(c.id),{method:'DELETE',body:JSON.stringify({revision:c.revision})});load()}catch(e){msg.textContent=e.message}}async function test(c,el){const out=el.querySelector('[data-result]');out.textContent='正在测试…';try{const d=await api('/api/cameras/'+encodeURIComponent(c.id)+'/test',{method:'POST'});out.textContent=d.ok?`连接成功 · ${d.width}×${d.height} · ${d.elapsed_ms} ms`:d.error}catch(e){out.textContent=e.message}}function preview(c,el){const img=el.querySelector('[data-preview]');img.style.display=img.style.display==='none'?'block':'none';img.src='/stream/'+encodeURIComponent(c.id)+'?t='+Date.now()}load();
</script>'''

class CloudSettingsUpdate(BaseModel):
    mqtt: dict | None = None
    s3: dict | None = None
    auto_upload: bool | None = None

class AutoUploadUpdate(BaseModel):
    enabled: bool

class CameraCreate(BaseModel):
    id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
    name: str = Field(min_length=1, max_length=200)
    main_url: str = Field(min_length=1, max_length=2048)
    sub_url: str | None = Field(default=None, max_length=2048)
    detection_fps: int = Field(default=3, ge=1, le=30)
    min_capture_score: float = Field(default=0.78, ge=0.0, le=1.0)
    enabled: bool = True

class CameraUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    main_url: str | None = Field(default=None, max_length=2048)
    sub_url: str | None = Field(default=None, max_length=2048)
    keep_substream: bool = True
    detection_fps: int = Field(default=3, ge=1, le=30)
    min_capture_score: float | None = Field(default=None, ge=0.0, le=1.0)
    enabled: bool = True
    revision: int = Field(ge=1)

class CameraRevision(BaseModel):
    revision: int = Field(ge=1)

class CameraEnabled(CameraRevision):
    enabled: bool

class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=256)

class PasswordChange(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=PASSWORD_MIN_LENGTH, max_length=256)

class UserCreate(BaseModel):
    username: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=PASSWORD_MIN_LENGTH, max_length=256)

class UserUpdate(BaseModel):
    username: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
    is_active: bool
    password: str | None = Field(default=None, min_length=PASSWORD_MIN_LENGTH, max_length=256)


def create_app(engine: RuntimeEngine) -> FastAPI:
    app = FastAPI(title="Edge Face Gateway", version="0.4.2")
    process = psutil.Process(os.getpid())
    logical_cpus = max(1, psutil.cpu_count(logical=True) or 1)
    sampler = SystemMetricsSampler(process, logical_cpus)
    app.add_event_handler("shutdown", sampler.stop)
    database = getattr(engine, "database", None)
    auth_service = AuthService(database) if database is not None else None
    if auth_service is not None:
        auth_service.ensure_initial_admin()
    app.state.auth_service = auth_service

    @app.middleware("http")
    async def require_authentication(request: Request, call_next):
        path = request.url.path
        if path in {"/login", "/api/auth/login"}:
            return await call_next(request)
        if auth_service is None:
            return JSONResponse({"detail": "authentication database unavailable"}, status_code=503)
        user = auth_service.user_for_session(request.cookies.get(SESSION_COOKIE))
        if user is None:
            if path.startswith(("/api/", "/media/", "/stream/")):
                return JSONResponse({"detail": "authentication required"}, status_code=401)
            return RedirectResponse("/login", status_code=303)
        request.state.user = user
        if user.must_change_password and path not in {
            "/change-password", "/api/auth/change-password", "/api/auth/logout",
        }:
            if path.startswith("/api/"):
                return JSONResponse({"detail": "password change required"}, status_code=428)
            return RedirectResponse("/change-password", status_code=303)
        origin = request.headers.get("origin")
        if origin and request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin_url = urlsplit(origin)
            if (origin_url.scheme.lower() != request.url.scheme.lower()
                    or origin_url.netloc.lower() != request.url.netloc.lower()):
                return JSONResponse({"detail": "invalid request origin"}, status_code=403)
        return await call_next(request)

    @app.get("/login", response_class=HTMLResponse)
    def login_page(request: Request):
        if auth_service is not None:
            user = auth_service.user_for_session(request.cookies.get(SESSION_COOKIE))
            if user is not None:
                target = "/change-password" if user.must_change_password else "/"
                return RedirectResponse(target, status_code=303)
        return LOGIN_TEMPLATE.replace("__BODY__", LOGIN_BODY).replace("__SCRIPT__", LOGIN_SCRIPT)

    @app.get("/change-password", response_class=HTMLResponse)
    def change_password_page():
        return _page(CHANGE_PASSWORD_BODY, CHANGE_PASSWORD_SCRIPT, "")

    @app.post("/api/auth/login")
    def login(body: LoginRequest, request: Request):
        if auth_service is None:
            raise HTTPException(503, "authentication database unavailable")
        user = auth_service.authenticate(body.username.strip(), body.password)
        if user is None:
            raise HTTPException(401, "用户名或密码错误")
        try:
            token = auth_service.create_session(user.id)
        except UserNotFoundError as exc:
            raise HTTPException(401, "用户名或密码错误") from exc
        response = JSONResponse({
            "username": user.username,
            "must_change_password": user.must_change_password,
        })
        response.set_cookie(
            SESSION_COOKIE, token, max_age=SESSION_TTL_MS // 1000, httponly=True,
            secure=request.url.scheme == "https", samesite="lax", path="/",
        )
        return response

    @app.post("/api/auth/logout")
    def logout(request: Request):
        if auth_service is not None:
            auth_service.revoke_session(request.cookies.get(SESSION_COOKIE))
        response = JSONResponse({"ok": True})
        response.delete_cookie(SESSION_COOKIE, path="/", httponly=True, samesite="lax")
        return response

    @app.post("/api/auth/change-password")
    def change_password(body: PasswordChange, request: Request):
        if auth_service is None:
            raise HTTPException(503, "authentication database unavailable")
        if body.current_password == body.new_password:
            raise HTTPException(422, "新密码必须与当前密码不同")
        try:
            token = auth_service.change_password(
                request.state.user.id, body.current_password, body.new_password,
            )
        except AuthenticationError as exc:
            raise HTTPException(401, "当前密码错误") from exc
        except UserNotFoundError as exc:
            raise HTTPException(401, "账号不可用") from exc
        response = JSONResponse({"ok": True})
        response.set_cookie(
            SESSION_COOKIE, token, max_age=SESSION_TTL_MS // 1000, httponly=True,
            secure=request.url.scheme == "https", samesite="lax", path="/",
        )
        return response

    @app.get("/users", response_class=HTMLResponse)
    def users_page():
        return _page(USERS_BODY, USERS_SCRIPT, "users")

    @app.get("/api/users")
    def list_users():
        if auth_service is None:
            raise HTTPException(503, "authentication database unavailable")
        return {"items": auth_service.list_users()}

    @app.post("/api/users", status_code=201)
    def create_user(body: UserCreate):
        if auth_service is None:
            raise HTTPException(503, "authentication database unavailable")
        try:
            return auth_service.create_user(body.username, body.password)
        except DuplicateUsernameError as exc:
            raise HTTPException(409, "用户名已存在") from exc

    @app.put("/api/users/{user_id}")
    def update_user(user_id: int, body: UserUpdate):
        if auth_service is None:
            raise HTTPException(503, "authentication database unavailable")
        try:
            return auth_service.update_user(user_id, body.username, body.is_active, body.password)
        except UserNotFoundError as exc:
            raise HTTPException(404, "用户不存在") from exc
        except DuplicateUsernameError as exc:
            raise HTTPException(409, "用户名已存在") from exc
        except LastActiveUserError as exc:
            raise HTTPException(409, "不能停用最后一个有效用户") from exc

    @app.delete("/api/users/{user_id}")
    def delete_user(user_id: int):
        if auth_service is None:
            raise HTTPException(503, "authentication database unavailable")
        try:
            auth_service.delete_user(user_id)
        except UserNotFoundError as exc:
            raise HTTPException(404, "用户不存在") from exc
        except LastActiveUserError as exc:
            raise HTTPException(409, "不能删除最后一个有效用户") from exc
        return {"ok": True}

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _page(HOME_BODY, HOME_SCRIPT, "home")

    @app.get("/album", response_class=HTMLResponse)
    def album():
        return _page(ALBUM_BODY, ALBUM_SCRIPT, "album")

    @app.get("/logs", response_class=HTMLResponse)
    def logs_page():
        return _page(LOGS_BODY, LOGS_SCRIPT, "logs")

    @app.get("/system-logs", response_class=HTMLResponse)
    def system_logs_page():
        return _page(SYSTEM_LOGS_BODY, SYSTEM_LOGS_SCRIPT, "system-logs")

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page():
        return _page(SETTINGS_BODY, SETTINGS_SCRIPT, "settings")

    @app.get("/cameras", response_class=HTMLResponse)
    def cameras_page():
        return _page(CAMERAS_BODY, CAMERAS_SCRIPT, "cameras")

    def _camera_row(row, status_row=None):
        runtime = engine.get_runtime(row.id)
        runtime_status = "STOPPED"
        if runtime is not None:
            runtime_status = "ONLINE" if runtime.main.connected and (runtime.sub is None or runtime.sub.connected) else ("CONNECTING" if runtime.main.thread and runtime.main.thread.is_alive() else "ERROR")
        camera_config = json.loads(row.config_json or "{}")
        return {"id": row.id, "name": row.name, "enabled": row.enabled, "stream_mode": row.stream_mode,
                "detection_fps": row.detection_fps,
                "min_capture_score": float(camera_config.get("min_capture_score", 0.78)),
                "revision": row.revision, "has_main_url": bool(row.main_url_encrypted),
                "runtime_status": runtime_status}

    @app.get("/api/system-settings")
    def get_system_settings():
        if getattr(engine, "cloud_services", None) is None:
            return {"mqtt": {"enabled": False}, "s3": {"enabled": False}, "auto_upload": False}
        return engine.cloud_services.public_config()

    @app.put("/api/system-settings")
    def update_system_settings(body: CloudSettingsUpdate):
        if getattr(engine, "cloud_services", None) is None:
            raise HTTPException(503, "cloud services unavailable")
        payload = body.model_dump(exclude_none=True)
        mqtt, s3 = payload.get("mqtt", {}), payload.get("s3", {})
        if mqtt.get("enabled") and (not mqtt.get("host") or not mqtt.get("gateway_id") or not mqtt.get("client_id")):
            raise HTTPException(422, "MQTT Broker、Client ID 和 Gateway ID 不能为空")
        if mqtt.get("gateway_id") is not None and not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", str(mqtt["gateway_id"])):
            raise HTTPException(422, "Gateway ID 仅允许字母、数字、下划线、点和连字符")
        if mqtt.get("client_id") is not None and not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", str(mqtt["client_id"])):
            raise HTTPException(422, "MQTT Client ID 格式无效")
        if mqtt.get("port") is not None and not 1 <= int(mqtt["port"]) <= 65535:
            raise HTTPException(422, "MQTT 端口无效")
        if mqtt.get("qos") is not None and int(mqtt["qos"]) not in (0, 1, 2):
            raise HTTPException(422, "MQTT QoS 必须为 0、1 或 2")
        if mqtt.get("topic_prefix") is not None and (not str(mqtt["topic_prefix"]).strip("/") or "#" in str(mqtt["topic_prefix"]) or "+" in str(mqtt["topic_prefix"])):
            raise HTTPException(422, "MQTT Topic 前缀无效")
        if s3.get("enabled") and not s3.get("bucket"):
            raise HTTPException(422, "启用 S3 时 Bucket 不能为空")
        if s3.get("endpoint_url") and not str(s3["endpoint_url"]).startswith(("http://", "https://")):
            raise HTTPException(422, "S3 Endpoint 必须以 http:// 或 https:// 开头")
        try:
            result = engine.cloud_services.save(payload)
            log.info("cloud service settings updated mqtt=%s s3=%s auto_upload=%s",
                     result["mqtt"]["enabled"], result["s3"]["enabled"], result["auto_upload"],
                     extra={"category": "SYSTEM"})
            return result
        except Exception as exc:
            log.exception("failed to save cloud service settings")
            raise HTTPException(400, "保存云服务配置失败") from exc

    @app.put("/api/system-settings/auto-upload")
    def set_auto_upload(body: AutoUploadUpdate):
        if getattr(engine, "cloud_services", None) is None:
            raise HTTPException(503, "cloud services unavailable")
        try:
            current = engine.cloud_services.public_config()
            if body.enabled and not current["s3"].get("enabled"):
                raise HTTPException(409, "请先在系统设置中启用并配置 S3 服务")
            return engine.cloud_services.save({"auto_upload": body.enabled})
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(400, "保存自动上传设置失败") from exc

    @app.post("/api/system-settings/test-mqtt")
    def test_mqtt():
        if getattr(engine, "cloud_services", None) is None:
            raise HTTPException(503, "cloud services unavailable")
        return engine.cloud_services.test_mqtt()

    @app.get("/api/system-settings/mqtt-status")
    def mqtt_status():
        if getattr(engine, "cloud_services", None) is None:
            raise HTTPException(503, "cloud services unavailable")
        return engine.cloud_services.mqtt_status()

    @app.post("/api/system-settings/test-s3")
    def test_s3():
        if getattr(engine, "cloud_services", None) is None:
            raise HTTPException(503, "cloud services unavailable")
        return engine.cloud_services.test_s3()

    @app.post("/api/captures/{image_id}/upload")
    def upload_capture(image_id: int):
        if getattr(engine, "cloud_services", None) is None:
            raise HTTPException(503, "cloud services unavailable")
        try:
            if not engine.cloud_services.public_config()["s3"].get("enabled"):
                raise HTTPException(409, "请先在系统设置中启用 S3 服务")
            result = engine.cloud_services.upload_image(image_id)
            log.info("capture uploaded image_id=%s key=%s", image_id, result["key"], extra={"category": "STORAGE"})
            return result
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            log.exception("capture cloud upload failed image_id=%s", image_id)
            raise HTTPException(502, "云存储上传失败，请检查配置及 Bucket 权限") from exc

    @app.get("/api/cameras")
    def list_cameras(page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=100)):
        if engine.database is None:
            return {"items": [], "pagination": {"page": page, "page_size": page_size, "total": 0}}
        with engine.database.sessions() as session:
            result = paginate(session, Camera, Page(page, page_size), order_by=(Camera.id,))
            items = []
            for row in result["items"]:
                st = session.get(CameraStatus, row.id)
                items.append(_camera_row(row, st))
            return {"items": items, "pagination": result["pagination"]}

    @app.post("/api/cameras", status_code=201)
    def create_camera(body: CameraCreate):
        if engine.database is None or engine.camera_repository is None:
            raise HTTPException(503, "camera persistence unavailable")
        if not body.name.strip(): raise HTTPException(422, "摄像头名称不能为空")
        if not body.main_url.lower().startswith(("rtsp://", "rtsps://")) or (body.sub_url and not body.sub_url.lower().startswith(("rtsp://", "rtsps://"))):
            raise HTTPException(422, "视频源必须使用 rtsp:// 或 rtsps://")
        try:
            with engine.database.sessions.begin() as session:
                if engine.camera_repository.get(session, body.id):
                    raise HTTPException(409, "摄像头 ID 已存在")
                row = engine.camera_repository.create(session, camera_id=body.id, name=body.name.strip(),
                    main_url=body.main_url, sub_url=body.sub_url, detection_fps=body.detection_fps,
                    min_capture_score=body.min_capture_score, enabled=body.enabled)
                result = _camera_row(row)
            engine.add_or_replace_camera(CameraConfig(
                body.id, body.name.strip(), body.main_url, body.sub_url, body.enabled,
                body.detection_fps, body.min_capture_score,
            ))
            log.info("camera created camera_id=%s enabled=%s stream_mode=%s", body.id, body.enabled, result["stream_mode"], extra={"category": "CAMERA", "camera_id": body.id})
            return result
        except HTTPException: raise
        except Exception as exc: raise HTTPException(400, "创建摄像头失败") from exc

    @app.put("/api/cameras/{camera_id}")
    def update_camera(camera_id: str, body: CameraUpdate):
        if engine.database is None or engine.camera_repository is None: raise HTTPException(503, "camera persistence unavailable")
        if not body.name.strip(): raise HTTPException(422, "摄像头名称不能为空")
        if body.main_url and not body.main_url.lower().startswith(("rtsp://", "rtsps://")): raise HTTPException(422, "主码流地址必须为 RTSP URL")
        if body.sub_url and not body.sub_url.lower().startswith(("rtsp://", "rtsps://")): raise HTTPException(422, "子码流地址必须为 RTSP URL")
        try:
            with engine.database.sessions.begin() as session:
                row = engine.camera_repository.get(session, camera_id)
                if row is None: raise HTTPException(404, "摄像头不存在")
                engine.camera_repository.update(session, row, name=body.name.strip(), main_url=body.main_url,
                    sub_url=body.sub_url, keep_sub=body.keep_substream, detection_fps=body.detection_fps,
                    min_capture_score=body.min_capture_score, enabled=body.enabled,
                    expected_revision=body.revision)
                main_url = engine.camera_repository.cipher.decrypt(row.main_url_encrypted)
                sub_url = engine.camera_repository.cipher.decrypt(row.sub_url_encrypted)
                camera_config = json.loads(row.config_json or "{}")
                cfg = CameraConfig(row.id, row.name, main_url, sub_url, row.enabled, row.detection_fps,
                                   float(camera_config.get("min_capture_score", 0.78)))
                result = _camera_row(row)
            engine.add_or_replace_camera(cfg)
            log.info("camera updated camera_id=%s revision=%s enabled=%s stream_mode=%s", camera_id, result["revision"], result["enabled"], result["stream_mode"], extra={"category": "CAMERA", "camera_id": camera_id})
            return result
        except RuntimeError as exc:
            if str(exc) == "revision_conflict": raise HTTPException(409, "配置已被其他请求修改，请刷新后重试")
            raise
        except HTTPException: raise
        except Exception as exc: raise HTTPException(400, "更新摄像头失败") from exc

    @app.put("/api/cameras/{camera_id}/enabled")
    def set_camera_enabled(camera_id: str, body: CameraEnabled):
        if engine.database is None or engine.camera_repository is None: raise HTTPException(503, "camera persistence unavailable")
        try:
            with engine.database.sessions.begin() as session:
                row = engine.camera_repository.get(session, camera_id)
                if row is None: raise HTTPException(404, "摄像头不存在")
                engine.camera_repository.set_enabled(session, row, body.enabled, body.revision)
                main_url = engine.camera_repository.cipher.decrypt(row.main_url_encrypted)
                sub_url = engine.camera_repository.cipher.decrypt(row.sub_url_encrypted)
                camera_config = json.loads(row.config_json or "{}")
                cfg = CameraConfig(row.id, row.name, main_url, sub_url, row.enabled, row.detection_fps,
                                   float(camera_config.get("min_capture_score", 0.78)))
            engine.add_or_replace_camera(cfg)
            log.info("camera enabled state changed camera_id=%s enabled=%s", camera_id, body.enabled, extra={"category": "CAMERA", "camera_id": camera_id})
            return {"ok": True, "enabled": body.enabled}
        except RuntimeError as exc:
            if str(exc) == "revision_conflict": raise HTTPException(409, "配置已变化，请刷新后重试")
            raise

    @app.delete("/api/cameras/{camera_id}")
    def delete_camera(camera_id: str, body: CameraRevision):
        if engine.database is None: raise HTTPException(503, "camera persistence unavailable")
        with engine.database.sessions.begin() as session:
            row = session.get(Camera, camera_id)
            if row is None: raise HTTPException(404, "摄像头不存在")
            if row.revision != body.revision: raise HTTPException(409, "配置已变化，请刷新后重试")
            session.delete(row)
        engine.remove_camera(camera_id)
        log.info("camera deleted camera_id=%s", camera_id, extra={"category": "CAMERA", "camera_id": camera_id})
        return {"ok": True}

    _probe_slots = threading.BoundedSemaphore(2)
    @app.post("/api/cameras/{camera_id}/test")
    def test_camera(camera_id: str):
        if engine.database is None or engine.camera_repository is None: raise HTTPException(503, "camera persistence unavailable")
        with engine.database.sessions() as session:
            row = engine.camera_repository.get(session, camera_id)
            if row is None: raise HTTPException(404, "摄像头不存在")
            url = engine.camera_repository.cipher.decrypt(row.main_url_encrypted)
        if not _probe_slots.acquire(blocking=False): raise HTTPException(429, "当前测试任务较多，请稍后重试")
        try:
            result = engine.test_stream(url)
            log.info("camera RTSP test camera_id=%s ok=%s", camera_id, result.get("ok"), extra={"category": "CAMERA", "camera_id": camera_id})
            return result
        finally: _probe_slots.release()

    @app.get("/api/status")
    def status():
        cameras = [r.metrics.snapshot(r.mode, r.main, r.sub) for r in engine.runtimes]
        cpu_raw, host_cpu = sampler.snapshot()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        return JSONResponse({"version": "0.4.2", "uptime_seconds": round(time.monotonic() - engine.host_started_mono, 1),
                             "model": os.getenv("FACE_MODEL_PATH", "configured model"),
                             "system": {"process_cpu_percent": round(cpu_raw / logical_cpus, 1),
                                        "process_cpu_cores": round(cpu_raw / 100.0, 2),
                                        "process_cpu_raw_percent": round(cpu_raw, 1),
                                        "host_cpu_percent": round(host_cpu, 1),
                                        "logical_cpus": logical_cpus,
                                        "process_rss_mb": round(memory_mb, 1)},
                             "cameras": cameras})

    @app.get("/api/captures")
    def captures(camera_id: str | None = None, kind: str = "all", offset: int = Query(0, ge=0),
                 limit: int = Query(48, ge=1, le=100), page: int | None = Query(None, ge=1),
                 page_size: int | None = Query(None, ge=1, le=100)):
        if kind not in {"all", "frame", "annotated", "face"}:
            raise HTTPException(status_code=400, detail="invalid image kind")
        allowed = {"frame": "原始帧", "annotated": "评分框", "face": "人脸裁剪"}
        if getattr(engine, "database", None) is not None:
            selected_page_size = page_size or limit
            selected_page = page or (offset // selected_page_size + 1)
            try: paging = Page(selected_page, selected_page_size)
            except ValueError as exc: raise HTTPException(status_code=400, detail=str(exc))
            with engine.database.sessions() as session:
                criteria = []
                if camera_id: criteria.append(CaptureImage.camera_id == camera_id)
                if kind != "all": criteria.append(CaptureImage.image_type == kind)
                result = paginate(session, CaptureImage, paging, *criteria,
                                  order_by=(desc(CaptureImage.captured_at_ms), desc(CaptureImage.id)))
                camera_ids = sorted(set(session.scalars(select(Camera.id)).all()) | set(session.scalars(select(CaptureImage.camera_id).distinct()).all()))
                items = []
                for row in result["items"]:
                    rel = Path(row.file_path)
                    cid, filename = rel.parent.name, rel.name
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(row.captured_at_ms / 1000))
                    items.append({"id": row.id, "cloud_upload_status": row.cloud_upload_status, "camera_id": row.camera_id, "kind": row.image_type,
                        "kind_label": allowed.get(row.image_type, row.image_type), "timestamp": timestamp,
                        "width": row.width, "height": row.height, "score": row.confidence,
                        "quality_score": row.quality_score,
                        "url": "/media/" + quote(cid) + "/" + quote(filename),
                        "thumbnail_url": "/media/" + quote(cid) + "/" + quote(filename) + "?thumbnail=true"})
                return {"total": result["pagination"]["total"], "offset": paging.offset,
                        "limit": paging.page_size, "page": result["pagination"]["page"],
                        "pagination": result["pagination"], "cameras": camera_ids, "items": items}

        candidates = []
        camera_ids = sorted(p.name for p in engine.output_dir.iterdir() if p.is_dir()) if engine.output_dir.exists() else []
        selected = [camera_id] if camera_id else camera_ids
        for cid in selected:
            if Path(cid).name != cid or cid not in camera_ids: continue
            for path in (engine.output_dir / cid).glob("*.jpg"):
                name = path.name
                item_kind = "frame" if name.endswith("_frame.jpg") else "annotated" if name.endswith("_detections.jpg") else "face" if "_face_" in name else None
                if item_kind and (kind == "all" or kind == item_kind): candidates.append((cid, path, item_kind))
        candidates.sort(key=lambda x: x[1].name, reverse=True)
        total = len(candidates); selected_page = page or (offset // (page_size or limit) + 1); selected_limit = page_size or limit
        page_items = candidates[offset:offset + selected_limit] if page is None else candidates[(selected_page-1)*selected_limit:selected_page*selected_limit]
        items = []
        for cid, path, item_kind in page_items:
            image = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)
            height, width = (image.shape[0] * 8, image.shape[1] * 8) if image is not None else (0, 0)
            score = None
            if item_kind == "face":
                try: score = float(path.name.split("_score_", 1)[1].removesuffix(".jpg"))
                except (IndexError, ValueError): pass
            items.append({"camera_id": cid, "kind": item_kind, "kind_label": allowed[item_kind],
                "timestamp": path.name.split("_face_", 1)[0].replace("T", " ").replace("Z", " UTC"),
                "width": width, "height": height, "score": score,
                "url": "/media/" + quote(cid) + "/" + quote(path.name),
                "thumbnail_url": "/media/" + quote(cid) + "/" + quote(path.name) + "?thumbnail=true"})
        return {"total": total, "offset": offset, "limit": selected_limit, "cameras": camera_ids, "items": items}

    @app.get("/media/{camera_id}/{filename}")
    def media(camera_id: str, filename: str, thumbnail: bool = False):
        if Path(camera_id).name != camera_id or Path(filename).name != filename or not filename.lower().endswith(".jpg"):
            raise HTTPException(status_code=400, detail="invalid media path")
        root = engine.output_dir.resolve()
        target = (root / camera_id / filename).resolve()
        if root not in target.parents or not target.is_file():
            raise HTTPException(status_code=404, detail="image not found")
        if not thumbnail:
            return FileResponse(target, media_type="image/jpeg")
        import cv2
        image = cv2.imread(str(target), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=404, detail="image could not be decoded")
        height, width = image.shape[:2]
        scale = min(1.0, 480.0 / max(width, height))
        if scale < 1.0:
            image = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))),
                               interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise HTTPException(status_code=500, detail="thumbnail encoding failed")
        return Response(content=encoded.tobytes(), media_type="image/jpeg")

    @app.get("/api/events")
    def events(event_type: str | None = None, camera_id: str | None = None,
               page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=100),
               limit: int | None = Query(None, ge=1, le=100)):
        if getattr(engine, "database", None) is not None:
            size = limit or page_size
            try: paging = Page(page, size)
            except ValueError as exc: raise HTTPException(status_code=400, detail=str(exc))
            with engine.database.sessions() as session:
                criteria = [BusinessEvent.event_type != "capture.manifest"]
                if event_type: criteria.append(BusinessEvent.event_type == event_type)
                if camera_id: criteria.append(BusinessEvent.camera_id == camera_id)
                result = paginate(session, BusinessEvent, paging, *criteria,
                                  order_by=(desc(BusinessEvent.timestamp_ms), desc(BusinessEvent.id)))
                camera_ids = sorted(set(session.scalars(select(Camera.id)).all()) | set(session.scalars(select(BusinessEvent.camera_id).where(BusinessEvent.camera_id.is_not(None)).distinct()).all()))
                rows = result["items"]
                items = [json.loads(row.payload_json) for row in rows]
                event_ids = [item["event_id"] for item in items if item.get("event_id")]
                outbox_by_id = {}
                if event_ids:
                    outbox_rows = session.scalars(
                        select(MqttOutboxEvent).where(MqttOutboxEvent.event_id.in_(event_ids))
                    ).all()
                    outbox_by_id = {row.event_id: row for row in outbox_rows}
                for row, item in zip(rows, items):
                    outbox = outbox_by_id.get(item.get("event_id"))
                    item["created_at_ms"] = row.timestamp_ms
                    item["published_at_ms"] = outbox.published_at_ms if outbox else None
                    item["push_status"] = (
                        "published" if outbox and outbox.published_at_ms is not None
                        else "pending" if outbox
                        else "not_queued"
                    )
                return {"total": result["pagination"]["total"], "items": items,
                        "pagination": result["pagination"], "cameras": camera_ids}
        path = getattr(getattr(engine, "event_log", None), "path", None)
        records = []
        if path and path.is_file():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    try: row = json.loads(line)
                    except json.JSONDecodeError: continue
                    kind = row.get("event_type") or "runtime.unknown"
                    row["event_type"] = kind
                    if kind == "capture.manifest" or (event_type and kind != event_type) or (camera_id and row.get("camera_id") != camera_id): continue
                    timestamp = row.get("timestamp_ms")
                    if not isinstance(timestamp, (int, float)):
                        try:
                            timestamp = int(datetime.fromisoformat(str(row.get("ts", "")).replace("Z", "+00:00")).timestamp() * 1000)
                        except (ValueError, TypeError):
                            timestamp = None
                    row["created_at_ms"] = timestamp
                    row["published_at_ms"] = None
                    row["push_status"] = "not_queued"
                    records.append(row)
        records.sort(key=lambda x: x.get("ts", ""), reverse=True)
        size = limit or page_size; start = (page - 1) * size
        camera_ids = sorted({r.cfg.camera_id for r in engine.runtimes})
        return {"total": len(records), "items": records[start:start+size],
                "pagination": {"page": page, "page_size": size, "total": len(records),
                    "total_pages": (len(records)+size-1)//size, "has_previous": page>1,
                    "has_next": page < (len(records)+size-1)//size}, "cameras": camera_ids}

    @app.get("/api/logs")
    def application_logs(level: str | None = None, category: str | None = None,
                         camera_id: str | None = None, page: int = Query(1, ge=1),
                         page_size: int = Query(50, ge=1, le=100)):
        if getattr(engine, "database", None) is None:
            return {"items": [], "pagination": {"page": page, "page_size": page_size, "total": 0,
                    "total_pages": 0, "has_previous": page > 1, "has_next": False}}
        with engine.database.sessions() as session:
            criteria = []
            if level: criteria.append(ApplicationLog.level == level.upper())
            if category: criteria.append(ApplicationLog.category == category.upper())
            if camera_id: criteria.append(ApplicationLog.camera_id == camera_id)
            result = paginate(session, ApplicationLog, Page(page, page_size), *criteria,
                              order_by=(desc(ApplicationLog.timestamp_ms), desc(ApplicationLog.id)))
            items = [{"id": row.id, "timestamp_ms": row.timestamp_ms, "level": row.level,
                "logger": row.logger, "category": row.category, "camera_id": row.camera_id,
                "message": row.message, "details": json.loads(row.details_json) if row.details_json else None,
                "exception": row.exception} for row in result["items"]]
            return {"items": items, "pagination": result["pagination"]}

    @app.get("/stream/{camera_id}")
    def stream(camera_id: str):
        runtime = engine.get_runtime(camera_id)
        if runtime is None:
            raise HTTPException(status_code=404, detail="camera not found")

        def frames() -> Iterator[bytes]:
            last = None
            while not engine.stop_event.is_set():
                jpeg = runtime.latest_preview
                if jpeg and jpeg is not last:
                    last = jpeg
                    yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
                time.sleep(0.05)
        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    return app
