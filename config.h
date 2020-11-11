#pragma once

#include <array>
#include <string>

namespace kaleido
{
    struct Keyframe
    {
        static constexpr float CONSTANT = 2e20f;

        float duration_seconds = 1;
        float mirror_internal_angle_degrees = CONSTANT;
        float mirror_bend_angle_degrees = CONSTANT;
        float background_texture_scale = CONSTANT;
        std::array<float, 2> background_texture_offset = { CONSTANT, CONSTANT };
        float background_texture_rotation_degrees = CONSTANT;
        float camera_position_polar_angle_degrees = CONSTANT;
        float medium_scattering_coefficient = CONSTANT;
    };

    struct Config
    {
        bool debug = false;
        bool three_mirrors = false;
        float mirror_internal_angle_degrees = 60;
        float mirror_bend_angle_degrees = 0;
        float mirror_albedo = 0.9f;
        float background_radiance = 1.0f;
        std::string background_texture_path;
        float background_texture_scale = 20;
        std::array<float, 2> background_texture_offset = { 0.5, 0.5 };
        float background_texture_rotation_degrees = 0;
        float medium_scattering_coefficient = 0;
        float medium_luminance_threshold = 0;
        float camera_field_of_view_degrees = 120;
        float camera_position_radius = 30;
        float camera_position_polar_angle_degrees = 0;
        float camera_position_azimuth_angle_degrees = 0;
        uint32_t stratifier_resolution = 32;
        uint32_t stratifier_rng_seed = 0;
        uint32_t nr_render_iterations = 1000;
        uint32_t max_trace_depth = 1000;
        uint32_t image_width = 256;
        uint32_t image_height = 256;
        bool animation_enabled = false;
        uint32_t animation_fps = 60;
        std::vector<Keyframe> animation_keyframes;
    };
}
