import sys
from pathlib import Path
import shutil
from zigzag.api import get_hardware_performance_zigzag
import socket
import json
import hashlib
import struct
import threading
from functools import lru_cache
from collections import OrderedDict
import pickle
import os
import hashlib
import functools
import atexit
import signal
import logging

from zigzag_template import workload_template,hardware_template,mapping_template

class PersistentLRUCache:
    """LRU cache with disk persistence"""
    
    def __init__(self, capacity=1000, cache_file='cache.pkl', auto_save_interval=60):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.cache_file = cache_file
        self.auto_save_interval = auto_save_interval
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.dirty = False
        self._save_timer = None
        
        # Load cache at startup
        self.load()
        
        # Start periodic saving
        if auto_save_interval > 0:
            self.auto_save()
        
        # Register multiple exit handlers
        self._register_exit_handlers()
    
    def _register_exit_handlers(self):
        """Register all possible exit handlers"""
        
        # Method 1: atexit - called on normal exit
        atexit.register(self._exit_save)
        
        # Method 2: signal - capture Ctrl+C (SIGINT) and kill (SIGTERM)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("[Cache] ✓ Exit handlers registered (atexit + signal)",flush=True)
    
    def _signal_handler(self, signum, frame):
        """Signal handler - captures Ctrl+C and other signals"""
        
        if signum == signal.SIGINT:
            signal_name = 'SIGINT' 
            print(f"\n[Cache] Received {signal_name}, saving cache...",flush=True)
            
            # Save cache
            self.save()
            
            # Print final stats
            stats = self.stats()
            print(f"[Cache] Final stats: {stats['hits']} hits, {stats['misses']} misses, "
                f"{stats['size']} entries",flush=True)
            
            # Exit program
            sys.exit(0)
    
    def _exit_save(self):
        """Save on exit (called by atexit)"""
        print("[Cache] Program exiting, saving cache...",flush=True)
        self.save()
    
    def get(self, key):
        """Get cached value"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            else:
                self.misses += 1
                return None
    
    def put(self, key, value):
        """Set cached value"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.cache[key] = value
            else:
                self.cache[key] = value
                
                if len(self.cache) > self.capacity:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
            
            self.dirty = True
    
    def load(self):
        """Load cache from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.cache = data.get('cache', OrderedDict())
                    self.hits = data.get('hits', 0)
                    self.misses = data.get('misses', 0)
                print(f"[Cache] ✓ Loaded {len(self.cache)} entries from {self.cache_file}",flush=True)
            except Exception as e:
                print(f"[Cache] ✗ Failed to load cache: {e}",flush=True)
                self.cache = OrderedDict()
        else:
            print(f"[Cache] No existing cache file, starting fresh",flush=True)
    
    def save(self):
        """Persist cache to disk"""
        if not self.dirty:
            print("[Cache] No changes to save",flush=True)
            return
        
        try:
            with self.lock:
                data = {
                    'cache': self.cache,
                    'hits': self.hits,
                    'misses': self.misses
                }
                
                # Ensure directory exists
                cache_dir = os.path.dirname(self.cache_file)
                if cache_dir and not os.path.exists(cache_dir):
                    os.makedirs(cache_dir)
                
                # Atomic write
                temp_file = self.cache_file + '.tmp'
                with open(temp_file, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                os.replace(temp_file, self.cache_file)
                self.dirty = False
                
                print(f"[Cache] ✓ Saved {len(self.cache)} entries to {self.cache_file}",flush=True)
        except Exception as e:
            print(f"[Cache] ✗ Failed to save cache: {e}",flush=True)
            import traceback
            traceback.print_exc()
    
    def auto_save(self):
        """Periodic auto-save"""
        def save_periodically():
            import time
            while True:
                time.sleep(self.auto_save_interval)
                if self.dirty:
                    print(f"[Cache] Auto-save triggered (interval: {self.auto_save_interval}s)",flush=True)
                    self.save()
        
        thread = threading.Thread(target=save_periodically, daemon=True, name='CacheAutoSave')
        thread.start()
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            self.dirty = True
            self.save()
    
    def stats(self):
        """Get statistics"""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0
            return {
                'hits': self.hits,
                'misses': self.misses,
                'size': len(self.cache),
                'capacity': self.capacity,
                'hit_rate': hit_rate,
                'dirty': self.dirty
            }
    
    def __len__(self):
        return len(self.cache)
    
    def __call__(self, func):
        """Decorator implementation"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Build cache key
            cache_key = self._make_key(func.__name__, args, kwargs)
            
            # Query cache
            cached_result = self.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            self.put(cache_key, result)
            
            return result
        
        # Attach helper methods
        wrapper.cache_info = self.stats
        wrapper.cache_clear = self.clear
        wrapper.cache_save = self.save
        
        return wrapper
    
    def _make_key(self, func_name, args, kwargs):
        """Generate cache key"""
        try:
            key_data = pickle.dumps((func_name, args, tuple(sorted(kwargs.items()))))
            return hashlib.md5(key_data).hexdigest()
        except:
            key_str = f"{func_name}:{str(args)}:{str(sorted(kwargs.items()))}"
            return hashlib.md5(key_str.encode()).hexdigest()
        

request_cache = PersistentLRUCache(
    capacity=1000000,
    cache_file='request_cache.pkl',
    auto_save_interval=60*60  # Auto-save every 60 seconds
)

def reset_logging_handlers(new_stream):
    """Reset all logging handlers output stream"""
    root_logger = logging.getLogger()
    
    # Iterate through all handlers
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            # Update handler's stream
            handler.stream = new_stream
            handler.flush()
    
    # Process all sub-loggers
    for name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler):
                handler.stream = new_stream
                handler.flush()

def process_request(params,process_dir):
    @request_cache
    def process(params):
        params=json.loads(params)
        core_type=params['arch']
        m=params['m']
        k=params['k']
        n=params['n']
        
        buffer=params['buf']*8
        mac=params['mac']


        workload = process_dir / 'gemm_layer.yaml'
        mapping = process_dir / f'mapping.yaml'
        accelerator = process_dir / f'hardware.yaml'

        workload_str = workload_template.format(m=m, k=k, n=n)
        hardware_str = hardware_template[core_type].format(buffer=buffer, mac=mac)
        mapping_str = mapping_template[core_type].format(*mac)

        with open(workload, 'w') as f:
            f.write(workload_str)
        with open(accelerator, 'w') as f:
            f.write(hardware_str)
        with open(mapping, 'w') as f:
            f.write(mapping_str)

        energy, latency, _ = get_hardware_performance_zigzag(workload=str(workload),
                                                            accelerator=str(accelerator),
                                                            mapping=str(mapping),
                                                            opt='EDP',
                                                            lpf_limit=4,
                                                            dump_folder=str(process_dir)
                                                            )

        return {'t':int(latency),'e':energy}
    return process(params)
    # import pickle
    # from zigzag.visualization.results.plot_cme import bar_plot_cost_model_evaluations_breakdown

    # # Load in the pickled list of CMEs
    # with open(output_dir/'list_of_cmes.pickle', 'rb') as fp:
    #     cme_for_all_layers = pickle.load(fp)

    # # Plot all the layers and save to 'plot_all.png'
    # bar_plot_cost_model_evaluations_breakdown(cme_for_all_layers, save_path=output_dir/"plot_breakdown.png") 

def recv_exactly(sock, n):
    """Receive exactly n bytes of data"""
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Socket connection broken")
        data += chunk
    return data

def handle_client(client_socket,address):
    """Handle a single client connection"""
    try:
        client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        request_count = 0
        output_dir = Path(__file__).parent / 'outputs' / str(address[1])
        output_dir.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                request_count += 1
                
                # 1. Receive request length
                length_data = recv_exactly(client_socket, 4)
                msg_length = struct.unpack('!I', length_data)[0]
                
                if msg_length == 0 or msg_length > 10 * 1024 * 1024:
                    print(f"[Error] Invalid length: {msg_length}",flush=True)
                    break
                
                # 2. Receive request data
                data = recv_exactly(client_socket, msg_length)
                
                # 3. Parse and process
                request = data.decode('utf-8')
                print(f"[Info] Request: {request}",flush=True)

                log_path=Path(__file__).parent / 'log' / f'{address[1]}.log'
                log_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    log_file=open(log_path,'w')
                    sys.stdout = log_file
                    sys.stderr = log_file
                    reset_logging_handlers(log_file)
                    response = process_request(request,output_dir)
                finally:
                    sys.stdout = sys.__stdout__
                    sys.stderr = sys.__stderr__
                    
                    if log_file:
                        log_file.close()

                print(f"[Info] Response: {response}",flush=True)
                
                # 4. Send response
                response_data = json.dumps(response).encode('utf-8')
                response_length = struct.pack('!I', len(response_data))
                full_response = response_length + response_data
                
                client_socket.sendall(full_response)
                log_path.unlink(missing_ok=True)
                
            except ConnectionError as e:
                shutil.rmtree(output_dir, ignore_errors=True)
                print(f"[Info] Connection error: {e}",flush=True)
                break
            except json.JSONDecodeError as e:
                print(f"[Error] JSON error: {e}",flush=True)
                break
            except Exception as e:
                print(f"[Error] Request processing error: {e}",flush=True)
                import traceback
                traceback.print_exc()
                break
        
    except Exception as e:
        print(f"[Error] Exception: {e}",flush=True)
        import traceback
        traceback.print_exc()
    finally:
        try:
            client_socket.close()
            print(f"[Info] Connection closed: {address}",flush=True)
        except:
            pass

def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('127.0.0.1', 18888))
    server_socket.listen(5)
    print("Python server listening on port 18888...",flush=True)
    
    while True:
        client_socket, address = server_socket.accept()
        print(f"[Info] Connection from {address}",flush=True)
        
        try:
            client_thread = threading.Thread(
                    target=handle_client,
                    args=(client_socket, address),
                    daemon=True  # Daemon thread, automatically ends when the main thread exits
                )
            client_thread.start()
            
        except KeyboardInterrupt:
            print("\n[Info] Server shutting down...",flush=True)
            break
        except Exception as e:
            print(f"[Error] Accept error: {e}",flush=True)

def main2():
    # Example request
    request = {
        'm': 128,
        'k': 2,
        'n': 33,
        'arch': 'os'
    }
    response = process_request(json.dumps(request))
    print(f"Response: {response}",flush=True)
if __name__ == '__main__':
    main()