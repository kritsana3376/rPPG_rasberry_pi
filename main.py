import numpy as np
from scipy import signal
import cv2
import time
import sys
import mediapipe as mp
import sys
import warnings
warnings.filterwarnings("ignore")


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def Face_Mesh(In_Img,face_mesh):
    ROI1 = np.zeros((10, 10, 3), np.uint8)
    ROI2 = np.zeros((10, 10, 3), np.uint8)

    results = face_mesh.process(In_Img)
    face_landmarks = results.multi_face_landmarks

    if not face_landmarks:
        cv2.putText(In_Img, "No face detected", (200, 200), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
    if (face_landmarks):
        POS = face_landmarks[0].landmark

        ROI1_X1 = POS[116].x * In_Img.shape[1]
        ROI1_Y1 = POS[116].y * In_Img.shape[0]
        ROI1_X2 = POS[203].x * In_Img.shape[1]
        ROI1_Y2 = POS[203].y * In_Img.shape[0]

        ROI2_X1 = POS[349].x * In_Img.shape[1]
        ROI2_Y1 = POS[349].y * In_Img.shape[0]
        ROI2_X2 = POS[376].x * In_Img.shape[1]
        ROI2_Y2 = POS[376].y * In_Img.shape[0]

        ROI1 = In_Img[int(ROI1_Y1):int(ROI1_Y2), int(ROI1_X1):int(ROI1_X2), :]
        ROI2 = In_Img[int(ROI2_Y1):int(ROI2_Y2), int(ROI2_X1):int(ROI2_X2), :]

        cv2.rectangle(In_Img, (int(ROI1_X1), int(ROI1_Y1)), (int(ROI1_X2), int(ROI1_Y2)), (0, 255, 0), 2)
        cv2.rectangle(In_Img, (int(ROI2_X1), int(ROI2_Y1)), (int(ROI2_X2), int(ROI2_Y2)), (0, 255, 0), 2)

    return ROI1, ROI2, In_Img





def main():
    cap = cv2.VideoCapture(0)
    face_mesh = mp.solutions.face_mesh.FaceMesh()
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    times = []
    data_buffer = []
    bpms = []
    frames_count = 0
    bpm = 0
    size_data = 30
    time_start = time.time()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ROI1, ROI2, frame = Face_Mesh(frame,face_mesh)

        green_ROI1 = np.mean(ROI1[:, :, 1])
        green_ROI2 = np.mean(ROI2[:, :, 1])
        green_avg = (green_ROI1 + green_ROI2) / 2

        buffer_size = 10 * round(size_data)  # Save 10 sec = 10*30 = 300 frames if 30fps

        # -> frames count
        frames_count = frames_count + 1  #
        time_in = time.time() - time_start
        data_buffer.append(green_avg)
        times.append(time_in)

        if (len(data_buffer) > buffer_size):  # data_buffer 10 sec
            start_t = round(fps)  # 30 fbs = 1 sec
            data_buffer = data_buffer[start_t:]  # 1 sec
            times = times[start_t:]

        # - > Convert to Array
        processed = np.array(data_buffer)
        Length_Data_buffer = len(data_buffer)

        # start calculating after the first 300 frames(10 sec) this video is 30 fps
        if len(data_buffer) == buffer_size:
            fps = float(Length_Data_buffer) / (times[-1] - times[
                0])  # calculate HR using a true fps of processor of the computer, not the fps the camera provide
            start_t = times[0]
            stop_t = times[-1]
            step_t = len(data_buffer)
            even_times = np.linspace(start_t, stop_t, step_t)
            # print(even_times)

            processed = signal.detrend(processed)  # detrend the signal to avoid interference of light change
            processed = butter_bandpass_filter(processed, 0.8, 3, fps, order=3)  # lowcut 0.8 Highcut 3
            interpolated = np.interp(even_times, times, processed)  # interpolation by 1
            interpolated = np.hamming(
                len(data_buffer)) * interpolated  # make the signal become more periodic (advoid spectral leakage)
            # norm = (interpolated - np.mean(interpolated))/np.std(interpolated)#normalization
            norm = interpolated / np.linalg.norm(interpolated)
            # print(norm)

            raw = np.fft.rfft(norm * 30)  # do real fft with the normalization multiplied by 10

            freqs = np.arange(len(data_buffer) / 2) * (float(fps) / len(data_buffer))
            freqs = 60. * freqs

            # Magnitude = 2.0*np.abs(fft_var[0:len_num_mag])

            fft = np.abs(raw) ** 2  # get amplitude spectrum

            idx = np.where((freqs > 50) & (freqs < 200))  # the range of frequency that HR is supposed to be within
            pruned = fft[idx]
            pfreq = freqs[idx]

            freqs = pfreq
            fft = pruned

            idx2 = np.argmax(pruned)  # max in the range can be HR

            bpm = freqs[idx2]
            bpms.append(bpm)
            # print(f"bpm =  {bpm}")

            # processed = butter_bandpass_filter(processed, 0.8, 3, fps, order=3) #lowcut 0.8 Highcut 3
            # ifft = np.fft.irfft(raw)
            # samples = processed  # multiply the signal with 5 for easier to see in the plot
            # TODO: find peaks to draw HR-like signal.

        # Display the resulting frame
        t_in = time.time()
        i = i + 1
        # print(f"count {i}")
        # print(t_in-t0)

        text = f"count:{i} fram, Time:{(time_in):.2f} s, FPS in bpm:{fps:.2f} fps, Data buffer:{Length_Data_buffer}, Buffer size:{buffer_size}, bpm:{bpm:.2f} bpm, bpm size:{len(bpms)}"
        sys.stdout.write("\r" + text)
        sys.stdout.flush()

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
