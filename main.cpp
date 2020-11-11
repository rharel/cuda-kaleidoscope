#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <time.h>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>

#include "external/include/json.hpp"

#include "config.h"
#include "cudu.h"
#include "filewatch.h"
#include "raytracer.h"

const char MAIN_WINDOW_TITLE[] = "Kaleidoscope";

namespace kaleido
{
    void from_json(const nlohmann::json& json, kaleido::Keyframe& keyframe)
    {
        keyframe.duration_seconds = json["duration_seconds"].get<float>();

        if (json.contains("mirror_internal_angle_degrees"))
        {
            keyframe.mirror_internal_angle_degrees = json["mirror_internal_angle_degrees"].get<float>();
        }
        if (json.contains("mirror_bend_angle_degrees"))
        {
            keyframe.mirror_bend_angle_degrees = json["mirror_bend_angle_degrees"].get<float>();
        }
        if (json.contains("background_texture_scale"))
        {
            keyframe.background_texture_scale = json["background_texture_scale"].get<float>();
        }
        if (json.contains("background_texture_offset"))
        {
            keyframe.background_texture_offset = json["background_texture_offset"].get<std::array<float, 2>>();
        }
        if (json.contains("background_texture_rotation_degrees"))
        {
            keyframe.background_texture_rotation_degrees = json["background_texture_rotation_degrees"].get<float>();
        }
        if (json.contains("camera_position_polar_angle_degrees"))
        {
            keyframe.camera_position_polar_angle_degrees = json["camera_position_polar_angle_degrees"].get<float>();
        }
        if (json.contains("medium_scattering_coefficient"))
        {
            keyframe.medium_scattering_coefficient = json["medium_scattering_coefficient"].get<float>();
        }
    }
}

kaleido::Config config_from_json_file(const std::string& path)
{
    nlohmann::json json_config;

    std::ifstream file_stream(path);
    if (!file_stream.is_open()) {
        throw std::runtime_error("failed to open configuration file");
    }
    file_stream >> json_config;

    kaleido::Config config;

    config.debug = json_config["debug"].get<bool>();
    config.three_mirrors = json_config["three_mirrors"].get<uint32_t>();
    config.mirror_internal_angle_degrees = json_config["mirror_internal_angle_degrees"].get<float>();
    config.mirror_bend_angle_degrees = json_config["mirror_bend_angle_degrees"].get<float>();
    config.mirror_albedo = json_config["mirror_albedo"].get<float>();
    config.background_radiance = json_config["background_radiance"].get<float>();
    config.background_texture_path = json_config["background_texture_path"].get<std::string>();
    config.background_texture_scale = json_config["background_texture_scale"].get<float>();
    config.background_texture_offset = json_config["background_texture_offset"].get<std::array<float, 2>>();
    config.background_texture_rotation_degrees = json_config["background_texture_rotation_degrees"].get<float>();
    config.medium_scattering_coefficient = json_config["medium_scattering_coefficient"].get<float>();
    config.medium_luminance_threshold = json_config["medium_luminance_threshold"].get<float>();
    config.camera_field_of_view_degrees = json_config["camera_field_of_view_degrees"].get<float>();
    config.camera_position_radius = json_config["camera_position_radius"].get<float>();
    config.camera_position_polar_angle_degrees = json_config["camera_position_polar_angle_degrees"].get<float>();
    config.camera_position_azimuth_angle_degrees = json_config["camera_position_azimuth_angle_degrees"].get<float>();
    config.stratifier_resolution = json_config["stratifier_resolution"].get<uint32_t>();
    config.stratifier_rng_seed = json_config["stratifier_rng_seed"].get<uint32_t>();
    config.nr_render_iterations = json_config["nr_render_iterations"].get<uint32_t>();
    config.max_trace_depth = json_config["max_trace_depth"].get<uint32_t>();
    config.image_width = json_config["image_width"].get<uint32_t>();
    config.image_height = json_config["image_height"].get<uint32_t>();
    config.animation_enabled = json_config["animation_enabled"].get<bool>();
    config.animation_fps = json_config["animation_fps"].get<uint32_t>();
    config.animation_keyframes = json_config["animation_keyframes"].get<std::vector<kaleido::Keyframe>>();

    return config;
}

cudu::host::Array3D<unsigned char> generated_image(const kaleido::Config& config)
{
    cudu::host::Array3D<unsigned char> image = kaleido::raytraced_image(config);

    const cv::Mat image_cv(
        int(image.shape()[0]),
        int(image.shape()[1]),
        CV_8UC3,
        image.begin()
    );
    cv::namedWindow(MAIN_WINDOW_TITLE, CV_WINDOW_AUTOSIZE);
    cv::imshow(MAIN_WINDOW_TITLE, image_cv);
    cv::waitKey(30);

    return image;
}

void save_image(const cudu::host::Array3D<unsigned char>& image)
{
    const time_t time = std::time(nullptr);
    tm local_time;
    localtime_s(&local_time, &time);

    std::ostringstream oss;
    oss << std::put_time(&local_time, "%Y_%m_%d_%H_%M_%S");
    const std::string time_str = oss.str();

    const std::string file_name = "world_" + time_str + ".png";

    const cv::Mat image_cv(
        int(image.shape()[0]),
        int(image.shape()[1]),
        CV_8UC3,
        const_cast<unsigned char*>(image.begin())
    );
    cv::imwrite(file_name, image_cv);

    std::cout << "Saved " + file_name << std::endl;
}

void run_gui_main_loop(
    std::atomic_bool& render_request,
    kaleido::Config& config,
    cudu::host::Array3D<unsigned char>& image)
{
    constexpr char KEYCODE_ESCAPE = 27;
    constexpr char KEYCODE_SAVE = 'p';
    constexpr char KEYCODE_TURN_UP = 'w';
    constexpr char KEYCODE_TURN_LEFT = 'a';
    constexpr char KEYCODE_TURN_DOWN = 's';
    constexpr char KEYCODE_TURN_RIGHT = 'd';
    constexpr char KEYCODE_ITERATION_INCREASE = '=';
    constexpr char KEYCODE_ITERATION_DECREASE = '-';
    constexpr char KEYCODE_STRATA_INCREASE = ']';
    constexpr char KEYCODE_STRATA_DECREASE = '[';

    char keycode = 0;

    while (cv::getWindowProperty(MAIN_WINDOW_TITLE, 0) >= 0 &&
           keycode != KEYCODE_ESCAPE)
    {
        switch (keycode = cv::waitKey(30))
        {
            case KEYCODE_SAVE: 
                save_image(image); break;
            case KEYCODE_TURN_UP: 
                config.camera_position_polar_angle_degrees += 5; 
                render_request = true; 
                break;
            case KEYCODE_TURN_DOWN: 
                config.camera_position_polar_angle_degrees -= 5; 
                render_request = true; 
                break;
            case KEYCODE_TURN_LEFT: 
                config.camera_position_azimuth_angle_degrees += 5; 
                render_request = true; 
                break;
            case KEYCODE_TURN_RIGHT: 
                config.camera_position_azimuth_angle_degrees -= 5; 
                render_request = true; 
                break;
            case KEYCODE_ITERATION_INCREASE: 
                config.nr_render_iterations *= 2; 
                render_request = true; 
                break;
            case KEYCODE_ITERATION_DECREASE: 
                config.nr_render_iterations = std::max(config.nr_render_iterations / 2, 2u); 
                render_request = true; 
                break;
            case KEYCODE_STRATA_INCREASE:
                config.stratifier_resolution *= 2;
                render_request = true;
                break;
            case KEYCODE_STRATA_DECREASE:
                config.stratifier_resolution = std::max(config.stratifier_resolution / 2, 1u);
                render_request = true;
                break;
            default: break;
        }
    }
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " CONFIG_FILE" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string config_path(argv[1]);
    auto config = config_from_json_file(config_path);
    cudu::host::Array3D<unsigned char> image = generated_image(config);
    
    if (config.animation_enabled)
    {
        size_t nr_frames_total = 0;
        size_t i_frame = 0;
        float frame_time_sum_s = 0;
        float frame_time_average_s = 0;

        cv::VideoWriter video_writer(
            "kaleidoscope.avi",
            cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
            config.animation_fps,
            cv::Size(config.image_width, config.image_height)
        );
        for (const kaleido::Keyframe& keyframe : config.animation_keyframes)
        {
            nr_frames_total += size_t(roundf(keyframe.duration_seconds * config.animation_fps));
        }
        for (const kaleido::Keyframe& keyframe : config.animation_keyframes)
        {
            const size_t nr_frames = size_t(roundf(config.animation_fps * keyframe.duration_seconds));
            kaleido::Keyframe step;
            #define KALEIDO_KEYFRAME_STEP(field) \
                step. ## field = keyframe. ## field == kaleido::Keyframe::CONSTANT ? 0 : \
                (keyframe. ## field - config. ## field) / nr_frames;
            KALEIDO_KEYFRAME_STEP(mirror_internal_angle_degrees);
            KALEIDO_KEYFRAME_STEP(mirror_bend_angle_degrees);
            KALEIDO_KEYFRAME_STEP(background_texture_scale);
            KALEIDO_KEYFRAME_STEP(background_texture_offset[0]);
            KALEIDO_KEYFRAME_STEP(background_texture_offset[1]);
            KALEIDO_KEYFRAME_STEP(background_texture_rotation_degrees);
            KALEIDO_KEYFRAME_STEP(camera_position_polar_angle_degrees);
            KALEIDO_KEYFRAME_STEP(medium_scattering_coefficient);
            #undef KALEIDO_KEYFRAME_STEP
            
            for (size_t i = 0; i < nr_frames; ++i)
            {
                auto start_clock = std::chrono::high_resolution_clock::now();
                
                i_frame += 1;
                const float nr_frames_remaining = nr_frames_total - i_frame;
                const uint32_t approx_time_remaining_s = round(frame_time_average_s * nr_frames_remaining);
                std::cout 
                    << "Rendering frame " << i_frame << "/" << nr_frames_total << " | "
                    << "Time remaining (approx): " << approx_time_remaining_s << " seconds" << std::endl;

                config.mirror_internal_angle_degrees += step.mirror_internal_angle_degrees;
                config.mirror_bend_angle_degrees += step.mirror_bend_angle_degrees;
                config.background_texture_scale += step.background_texture_scale;
                config.background_texture_offset[0] += step.background_texture_offset[0];
                config.background_texture_offset[1] += step.background_texture_offset[1];
                config.background_texture_rotation_degrees += step.background_texture_rotation_degrees;
                config.camera_position_polar_angle_degrees += step.camera_position_polar_angle_degrees;
                config.medium_scattering_coefficient += step.medium_scattering_coefficient;

                image = generated_image(config);
                const cv::Mat image_cv(
                    int(image.shape()[0]),
                    int(image.shape()[1]),
                    CV_8UC3,
                    image.begin()
                );
                video_writer << image_cv;
                
                auto end_clock = std::chrono::high_resolution_clock::now();
                
                frame_time_sum_s += std::chrono::duration_cast<std::chrono::milliseconds>(end_clock - start_clock).count() / 1000.0f;
                frame_time_average_s = frame_time_sum_s / i_frame;
            }
        }
        return EXIT_SUCCESS;
    }

    std::atomic_bool render_thread_stop = false;
    std::atomic_bool render_request = false;
    std::thread render_thread([&render_thread_stop, &render_request, &config, &image]() {
        while (!render_thread_stop)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            
            if (render_request)
            {
                std::cout << "Generating image..." << std::endl;
                image = generated_image(config);
                render_request = false;
                std::cout << "Ready" << std::endl;
            }
        }
    });

    filewatch::FileWatcher config_watcher(
        config_path,
        std::chrono::milliseconds(1000)
    );
    const auto on_config_file_change = [&render_request, &config, &image](
        const std::string& path,
        const filewatch::EventKind event_kind)
    {
        switch (event_kind) {
            case filewatch::EventKind::Created:
            case filewatch::EventKind::Modified: {
                config = config_from_json_file(path);
                render_request = true;
                break;
            }
            default: break;
        }
    };
    std::thread config_watcher_thread([&config_watcher, &on_config_file_change]() {
        config_watcher.start(on_config_file_change);
    });

    run_gui_main_loop(render_request, config, image);

    config_watcher.stop();
    config_watcher_thread.join();
    
    render_thread_stop = true;
    render_thread.join();

    return EXIT_SUCCESS;
}
