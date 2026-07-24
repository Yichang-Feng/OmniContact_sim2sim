import socket
import struct
import time

def check_udp():
    print("==================================================")
    print("🔍 [网络层诊断] 正在监听网卡 enx6c1ff724495a (192.168.123.123) 上的 RTPS 流量...")
    print("==================================================")
    
    ports = [7400, 7401, 7402]
    sockets = []
    local_ip = "192.168.123.123"
    mcast_grp = "239.255.0.1"
    
    for port in ports:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except AttributeError:
                pass
            s.bind(('0.0.0.0', port))
            
            # 加入组播组 239.255.0.1
            try:
                mreq = struct.pack("4s4s", socket.inet_aton(mcast_grp), socket.inet_aton(local_ip))
                s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
                print(f" 成功绑定端口 {port} 并加入组播组 {mcast_grp} (网卡 {local_ip})")
            except Exception as e:
                print(f" 端口 {port} 加入组播组失败: {e}")
                
            s.settimeout(0.1)
            sockets.append((port, s))
        except Exception as e:
            print(f" 绑定端口 {port} 失败: {e}")
            
    print("\n⏳ 正在监听 5 秒钟内的所有进入数据包...")
    start_time = time.time()
    pkt_count = 0
    
    while time.time() - start_time < 5.0:
        for port, s in sockets:
            try:
                data, addr = s.recvfrom(65536)
                pkt_count += 1
                if pkt_count <= 15:
                    print(f"📦 [收到数据包 #{pkt_count}] 来自 {addr[0]}:{addr[1]} -> 端口 {port} | 长度: {len(data)} 字节 | 前16字节: {data[:16].hex()}")
            except socket.timeout:
                pass
            except Exception as e:
                pass
                
    print(f"\n📊 [诊断统计]: 5秒内总计收到 {pkt_count} 个 UDP 数据包。")
    for _, s in sockets:
        s.close()

if __name__ == "__main__":
    check_udp()
