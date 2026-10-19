"""
M8 演示：容器化与部署实验室

运行:
    cd backend
    python -m knowflow.demos.m8_deploy_lab

本演示覆盖:
  1. Dockerfile 解析     —— 每条指令是干什么的
  2. docker-compose 解析  —— 服务编排、依赖顺序、持久化
  3. 云映射               —— 本地组件 <-> AWS/GCP/Azure
  4. Linux 运维命令       —— 常用排查技能
  5. 构建优化             —— 分层缓存、多阶段构建
  6. 部署清单             —— 上生产前的 checklist

环境说明: 本机未装 Docker。这个 demo 解释配置文件在做什么，
          真正的 docker compose up 需要装 Docker Desktop。
          能讲清楚配置和原理，比会敲命令更有价值。
"""

from __future__ import annotations

import pathlib


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


# ---------------------------------------------------------------------------
# 1. Dockerfile 解析
# ---------------------------------------------------------------------------
def demo_dockerfile():
    section("1. Dockerfile 解析 —— 每条指令在干什么")

    dockerfile = pathlib.Path(__file__).parent.parent.parent / "Dockerfile"
    if not dockerfile.exists():
        print("  [跳过] Dockerfile 不存在")
        return

    lines = dockerfile.read_text(encoding="utf-8").splitlines()
    key_instructions = {
        "FROM": "基础镜像（选 Python 3.12 slim，体积小）",
        "WORKDIR": "工作目录（所有后续命令在这里执行）",
        "COPY": "拷贝文件进镜像",
        "RUN": "执行命令（安装依赖等）",
        "ENV": "环境变量",
        "EXPOSE": "声明端口（不自动开放，只是文档）",
        "USER": "以什么用户运行（非 root 更安全）",
        "HEALTHCHECK": "健康检查（docker/K8s 用它判断服务是否可用）",
        "CMD": "容器启动命令（一个镜像只有一个 CMD）",
    }

    print(f"  文件: {dockerfile}")
    print(f"  共 {len(lines)} 行")
    print(f"\n  关键指令:")
    seen = set()
    for line in lines:
        line = line.strip()
        for keyword, desc in key_instructions.items():
            if line.startswith(keyword) and keyword not in seen:
                seen.add(keyword)
                print(f"    {keyword:<12} {desc}")
                break

    print(f"\n  两个阶段（多阶段构建）:")
    in_stage = False
    for line in lines:
        if line.strip().startswith("FROM"):
            stage = line.strip().split()[-1] if "AS" in line.upper() else "runtime"
            print(f"    {line.strip()[:60]}")
            in_stage = True

    print(f"\n  [OK] 多阶段构建：编译工具链不进最终镜像，体积从 1.2GB 缩到 ~300MB")


# ---------------------------------------------------------------------------
# 2. docker-compose 解析
# ---------------------------------------------------------------------------
def demo_compose():
    section("2. docker-compose 解析 —— 一键起全栈")

    compose = pathlib.Path(__file__).parent.parent.parent / "docker-compose.yml"
    if not compose.exists():
        print("  [跳过] docker-compose.yml 不存在")
        return

    content = compose.read_text(encoding="utf-8")

    # 提取服务名
    services = []
    in_services = False
    for line in content.splitlines():
        if line.strip() == "services:":
            in_services = True
            continue
        if in_services and line.startswith("  ") and not line.startswith("   "):
            name = line.strip().rstrip(":")
            if name and not name.startswith("#"):
                services.append(name)

    print(f"  服务清单 ({len(services)} 个):")
    service_descs = {
        "api": "FastAPI 服务（端口 8000）",
        "worker": "文档异步入库消费者",
        "db": "PostgreSQL 16（业务数据）",
        "redis": "Redis 7（缓存/限流）",
        "kafka": "Kafka（消息队列）",
        "qdrant": "Qdrant（向量数据库）",
    }
    for s in services:
        desc = service_descs.get(s, "")
        print(f"    {s:<10} {desc}")

    print(f"\n  关键配置:")
    if "depends_on" in content:
        print("    depends_on     启动顺序（api 等 db/redis 就绪才起）")
    if "healthcheck" in content:
        print("    healthcheck    保证服务真正可用（不只是进程起了）")
    if "volumes:" in content.split("services:")[-1]:
        print("    volumes        持久化数据（容器删了数据还在）")
    if "restart:" in content:
        print("    restart        挂了自动重启（unless-stopped）")

    print(f"\n  常用命令:")
    print(f"    docker compose up -d          后台起全栈")
    print(f"    docker compose logs -f api    看 API 日志")
    print(f"    docker compose down           一键全关")
    print(f"    docker compose ps             看服务状态")

    print(f"\n  [OK] 一条命令起 6 个服务，一条命令起 6 个服务")


# ---------------------------------------------------------------------------
# 3. 云映射
# ---------------------------------------------------------------------------
def demo_cloud_mapping():
    section("3. 云映射 —— 本地组件怎么搬到云上")

    rows = [
        ("本地组件", "AWS", "GCP", "Azure"),
        ("Docker 容器", "ECS / EKS", "Cloud Run / GKE", "Container Instances / AKS"),
        ("PostgreSQL", "RDS", "Cloud SQL", "Azure Database for PG"),
        ("Redis", "ElastiCache", "Memorystore", "Azure Cache for Redis"),
        ("Kafka", "MSK", "Pub/Sub", "Event Hubs"),
        ("对象存储", "S3", "Cloud Storage", "Blob Storage"),
        ("日志监控", "CloudWatch", "Cloud Logging", "Azure Monitor"),
    ]
    for r in rows:
        print(f"  {r[0]:<16} | {r[1]:<18} | {r[2]:<18} | {r[3]}")

    print(f"\n  核心思想:")
    print(f"    「代码零改动，换配置——容器跑 ECS，数据库用 RDS，Redis 用 ElastiCache，")
    print(f"      Kafka 用 MSK。Docker 镜像在任何云上都一样，这是容器化的价值。」")
    print(f"    [OK] Docker 的意义：本地=云上，不被云厂商锁死")


# ---------------------------------------------------------------------------
# 4. Linux 运维
# ---------------------------------------------------------------------------
def demo_linux():
    section("4. Linux 运维 —— 自查常考")

    commands = [
        ("进程管理", [
            ("ps aux | grep uvicorn", "找 uvicorn 进程"),
            ("kill -9 <PID>", "强杀进程"),
            ("nohup python -m uvicorn ... &", "后台运行"),
        ]),
        ("端口管理", [
            ("netstat -tlnp | grep 8000", "谁占了 8000 端口"),
            ("lsof -i :8000", "同上（macOS/Linux）"),
        ]),
        ("日志排查", [
            ("tail -f app.log", "实时看日志"),
            ('grep "ERROR" app.log | tail -20', "找最近 20 条错误"),
            ("grep -c 'ERROR' app.log", "统计错误数"),
        ]),
        ("系统资源", [
            ("top / htop", "CPU/内存"),
            ("free -h", "内存用量"),
            ("df -h", "磁盘用量"),
        ]),
    ]
    for category, cmds in commands:
        print(f"\n  {category}:")
        for cmd, desc in cmds:
            print(f"    {cmd:<42} {desc}")

    print(f"\n  [OK] 线上出问题时的排查工具箱")


# ---------------------------------------------------------------------------
# 5. 构建优化
# ---------------------------------------------------------------------------
def demo_build_optimization():
    section("5. 构建优化 —— 分层缓存 + 多阶段")

    print("  分层缓存（Dockerfile 的顺序很重要）:")
    print()
    print("    COPY pyproject.toml .     # 1. 先拷依赖文件")
    print("    RUN pip install ...       # 2. 装依赖（缓存这层）")
    print("    COPY . .                  # 3. 最后拷代码")
    print()
    print("    好处: 改代码后重新 build，依赖层命中缓存，不用重新装")
    print("    反例: COPY . . 放前面 -> 任何代码改动都导致重新装依赖")

    print(f"\n  多阶段构建:")
    print(f"    阶段 1 (builder):  build-essential + pip install")
    print(f"    阶段 2 (runtime):  只拷 /install 目录")
    print(f"    结果: gcc/头文件不进最终镜像，体积从 1.2GB -> 300MB")

    print(f"\n  其他优化:")
    print(f"    .dockerignore  排除 .git/venv/__pycache__，减小构建上下文")
    print(f"    python:3.12-slim  比 python:3.12 小 700MB")
    print(f"    --no-cache-dir    pip 不留缓存")

    print(f"\n  [OK] 镜像小 = 启动快、传输快、攻击面小")


# ---------------------------------------------------------------------------
# 6. 部署清单
# ---------------------------------------------------------------------------
def demo_checklist():
    section("6. 上生产前的 Checklist")

    checks = [
        ("安全", [
            "Docker 容器用非 root 用户",
            "生产环境变量走 Secret Manager（不写 .env）",
            "API 增加认证（JWT / API Key）",
            "CORS 限制到具体域名",
            "数据库密码不硬编码",
        ]),
        ("可靠性", [
            "所有服务配 healthcheck",
            "restart: unless-stopped",
            "数据库定期备份",
            "日志集中收集（ELK / CloudWatch）",
            "监控告警（Prometheus + Grafana）",
        ]),
        ("性能", [
            "API 多 worker（uvicorn --workers 4）",
            "数据库连接池参数调优",
            "静态文件走 CDN",
            "开启 gzip 压缩",
        ]),
    ]
    for category, items in checks:
        print(f"\n  {category}:")
        for item in items:
            print(f"    [ ] {item}")

    print(f"\n  [OK] 这份清单 = 「我知道生产部署要考虑什么」")


# ---------------------------------------------------------------------------
async def main() -> None:
    print(__doc__)

    demo_dockerfile()
    demo_compose()
    demo_cloud_mapping()
    demo_linux()
    demo_build_optimization()
    demo_checklist()

    section("完成")
    print("  概念回顾:")
    print("    · Docker       镜像=模板，容器=实例，解决环境一致性")
    print("    · compose      一键起全栈（api+worker+db+redis+kafka+qdrant）")
    print("    · 多阶段构建    编译工具链不进最终镜像，体积 -70%")
    print("    · 云映射        代码零改动换配置（ECS/RDS/ElastiCache/MSK）")
    print("    · Linux        进程/端口/日志/资源 排查工具箱")
    print("    · Checklist    安全/可靠性/性能 三维度")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
