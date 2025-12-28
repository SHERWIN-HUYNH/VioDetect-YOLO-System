"""
Script trích xuất frame gốc từ video 4K để lấy tọa độ chính xác
Author: Vehicle Counting System
"""

import cv2
import os


def extract_frame_from_video(video_path, frame_number=0, output_path="frame_4k.png"):
    """
    Trích xuất 1 frame từ video ở độ phân giải gốc
    
    Args:
        video_path: Đường dẫn đến video
        frame_number: Số thứ tự frame cần lấy (0 = frame đầu tiên)
        output_path: Đường dẫn lưu ảnh output
    
    Returns:
        True nếu thành công, False nếu thất bại
    """
    
    # Mở video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Không thể mở video: {video_path}")
        return False
    
    # Lấy thông tin video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Thông tin video:")
    print(f"   - Độ phân giải: {width}x{height}")
    print(f"   - FPS: {fps}")
    print(f"   - Tổng số frame: {total_frames}")
    print(f"   - Thời lượng: {total_frames/fps:.2f} giây")
    
    # Kiểm tra frame_number hợp lệ
    if frame_number < 0 or frame_number >= total_frames:
        print(f"❌ Frame number {frame_number} không hợp lệ (0-{total_frames-1})")
        cap.release()
        return False
    
    # Nhảy đến frame cần lấy
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Đọc frame
    ret, frame = cap.read()
    
    if not ret:
        print(f"❌ Không thể đọc frame {frame_number}")
        cap.release()
        return False
    
    # Lưu frame ra file (PNG để giữ chất lượng 100%)
    cv2.imwrite(output_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    cap.release()
    
    print(f"\n✅ Đã trích xuất frame {frame_number} thành công!")
    print(f"   - Kích thước: {frame.shape[1]}x{frame.shape[0]}")
    print(f"   - Lưu tại: {output_path}")
    print(f"   - Dung lượng: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    return True


def extract_multiple_frames(video_path, frame_list, output_dir="frames"):
    """
    Trích xuất nhiều frame cùng lúc
    
    Args:
        video_path: Đường dẫn video
        frame_list: List các số frame cần lấy, ví dụ [0, 30, 60]
        output_dir: Thư mục lưu các frame
    """
    
    # Tạo thư mục nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Không thể mở video: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video: {width}x{height}")
    print(f"🎯 Đang trích xuất {len(frame_list)} frames...\n")
    
    success_count = 0
    
    for frame_num in frame_list:
        if frame_num < 0 or frame_num >= total_frames:
            print(f"⚠️  Frame {frame_num} vượt quá giới hạn, bỏ qua")
            continue
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            output_path = os.path.join(output_dir, f"frame_{frame_num:05d}.png")
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print(f"✅ Frame {frame_num} → {output_path}")
            success_count += 1
        else:
            print(f"❌ Không thể đọc frame {frame_num}")
    
    cap.release()
    print(f"\n🎉 Hoàn thành! Đã trích xuất {success_count}/{len(frame_list)} frames")


def extract_frame_at_time(video_path, time_seconds, output_path="frame_at_time.png"):
    """
    Trích xuất frame tại thời điểm cụ thể (giây)
    
    Args:
        video_path: Đường dẫn video
        time_seconds: Thời điểm cần lấy (giây), ví dụ 5.5 = giây thứ 5.5
        output_path: Đường dẫn lưu ảnh
    """
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Không thể mở video: {video_path}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    if time_seconds < 0 or time_seconds > duration:
        print(f"❌ Thời gian {time_seconds}s vượt quá video (0-{duration:.2f}s)")
        cap.release()
        return False
    
    # Tính frame number từ thời gian
    frame_number = int(time_seconds * fps)
    
    print(f"⏱️  Lấy frame tại giây thứ {time_seconds} (frame {frame_number})")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    
    if ret:
        cv2.imwrite(output_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f"✅ Đã lưu frame tại: {output_path}")
        cap.release()
        return True
    else:
        print(f"❌ Không thể đọc frame")
        cap.release()
        return False


# ==================== CÁCH SỬ DỤNG ====================

if __name__ == "__main__":
    
    VIDEO_PATH = "assets/video/test_video_720p.mp4"  # ← THAY ĐỔI ĐƯỜNG DẪN VIDEO CỦA BẠN
    
    print("="*60)
    print("🎬 TRÍCH XUẤT FRAME TỪ VIDEO 4K")
    print("="*60)
    print()
    
    print("📌 Cách 1: Lấy frame đầu tiên của video")
    extract_frame_from_video(
        video_path=VIDEO_PATH,
        frame_number=0,  # Frame đầu tiên
        output_path="frame_4k_first.png"
    )
    
    print("\n" + "="*60 + "\n")
    
    print("📌 Cách 2: Lấy frame tại giây thứ 5")
    extract_frame_at_time(
        video_path=VIDEO_PATH,
        time_seconds=5.0,  # Giây thứ 5
        output_path="frame_4k_at_5s.png"
    )
    
    print("\n" + "="*60 + "\n")
    
    print("📌 Cách 3: Lấy nhiều frame để so sánh")
    extract_multiple_frames(
        video_path=VIDEO_PATH,
        frame_list=[0, 30, 60, 90, 120],  # Frame 0, 30, 60, 90, 120
        output_dir="extracted_frames"
    )
    
    print("\n" + "="*60)
    print("✅ HOÀN THÀNH!")
    print("="*60)
    print("\n📝 BƯỚC TIẾP THEO:")
    print("1. Mở file ảnh PNG vừa tạo bằng phần mềm xem ảnh")
    print("2. Dùng công cụ đo tọa độ (xem hướng dẫn tiếp theo)")
    print("3. Ghi lại tọa độ chính xác cho config.py")
    print("="*60)