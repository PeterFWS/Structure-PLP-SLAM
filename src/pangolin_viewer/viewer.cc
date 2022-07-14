#include "pangolin_viewer/viewer.h"

#include "PLPSLAM/config.h"
/**
 * This file is part of Structure PLP-SLAM, originally from OpenVSLAM.
 *
 * Copyright 2022 DFKI (German Research Center for Artificial Intelligence)
 * Modified by Fangwen Shu <Fangwen.Shu@dfki.de>
 *
 * If you use this code, please cite the respective publications as
 * listed on the github repository.
 *
 * Structure PLP-SLAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Structure PLP-SLAM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Structure PLP-SLAM. If not, see <http://www.gnu.org/licenses/>.
 */

#include "PLPSLAM/system.h"
#include "PLPSLAM/camera/base.h"
#include "PLPSLAM/camera/perspective.h"
#include "PLPSLAM/publish/frame_publisher.h"
#include "PLPSLAM/publish/map_publisher.h"

#include <opencv2/highgui.hpp>
#include <random>

#include <spdlog/spdlog.h>

namespace pangolin_viewer
{

    viewer::viewer(const std::shared_ptr<PLPSLAM::config> &cfg, PLPSLAM::system *system,
                   const std::shared_ptr<PLPSLAM::publish::frame_publisher> &frame_publisher,
                   const std::shared_ptr<PLPSLAM::publish::map_publisher> &map_publisher)
        : system_(system), frame_publisher_(frame_publisher), map_publisher_(map_publisher),
          interval_ms_(1000.0f / cfg->yaml_node_["PangolinViewer.fps"].as<float>(30.0)),
          viewpoint_x_(cfg->yaml_node_["PangolinViewer.viewpoint_x"].as<float>(0.0)),
          viewpoint_y_(cfg->yaml_node_["PangolinViewer.viewpoint_y"].as<float>(-10.0)),
          viewpoint_z_(cfg->yaml_node_["PangolinViewer.viewpoint_z"].as<float>(-0.1)),
          viewpoint_f_(cfg->yaml_node_["PangolinViewer.viewpoint_f"].as<float>(2000.0)),
          keyfrm_size_(cfg->yaml_node_["PangolinViewer.keyframe_size"].as<float>(0.1)),
          keyfrm_line_width_(cfg->yaml_node_["PangolinViewer.keyframe_line_width"].as<unsigned int>(1)),
          graph_line_width_(cfg->yaml_node_["PangolinViewer.graph_line_width"].as<unsigned int>(1)),
          point_size_(cfg->yaml_node_["PangolinViewer.point_size"].as<unsigned int>(2)),
          camera_size_(cfg->yaml_node_["PangolinViewer.camera_size"].as<float>(0.15)),
          camera_line_width_(cfg->yaml_node_["PangolinViewer.camera_line_width"].as<unsigned int>(2)),
          cs_(cfg->yaml_node_["PangolinViewer.color_scheme"].as<std::string>("black")),
          mapping_mode_(system->mapping_module_is_enabled()),
          loop_detection_mode_(system->loop_detector_is_enabled())
    {
        // FW:
        load_configuration(_cfg_path);
    }

    void viewer::run()
    {
        is_terminated_ = false;

        // FW: ---------------------------Start-------Pangolin GUI: Map Viewer--------------------------------
        pangolin::CreateWindowAndBind(map_viewer_name_, 1024, 768);

        glEnable(GL_BLEND);
        // glEnable(GL_COLOR_MATERIAL);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // depth testing to be enabled for 3D mouse handler
        glEnable(GL_DEPTH_TEST);

        // setup camera renderer
        s_cam_ = std::unique_ptr<pangolin::OpenGlRenderState>(new pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(map_viewer_width_, map_viewer_height_, viewpoint_f_, viewpoint_f_,
                                       map_viewer_width_ / 2, map_viewer_height_ / 2, 0.1, 1e6),
            pangolin::ModelViewLookAt(viewpoint_x_, viewpoint_y_, viewpoint_z_, 0, 0, 0, 0.0, -1.0, 0.0)));

        // create map window
        pangolin::View &d_cam = pangolin::CreateDisplay()
                                    .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -map_viewer_width_ / map_viewer_height_)
                                    .SetHandler(new pangolin::Handler3D(*s_cam_));

        // create menu panel
        create_menu_panel();

        // FW: ---------------------------Done-------Pangolin GUI: Map Viewer--------------------------------

        // create frame window
        cv::namedWindow(frame_viewer_name_);

        pangolin::OpenGlMatrix gl_cam_pose_wc; // camera -> world
        gl_cam_pose_wc.SetIdentity();

        while (true)
        {
            // FW: clear color and depth buffer
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // 1. draw the map window

            // get current camera pose as OpenGL matrix
            gl_cam_pose_wc = get_current_cam_pose();

            // make the rendering camera follow to the current camera
            follow_camera(gl_cam_pose_wc);

            // set rendering state
            d_cam.Activate(*s_cam_);
            glClearColor(cs_.bg_.at(0), cs_.bg_.at(1), cs_.bg_.at(2), cs_.bg_.at(3));

            // draw horizontal grid
            draw_horizontal_grid();
            // draw the current camera frustum
            draw_current_cam_pose(gl_cam_pose_wc);
            // draw keyframes and graphs
            draw_keyframes();
            // draw landmarks
            draw_landmarks();

            // FW: draw landmark plane
            if (map_publisher_->seg_or_not() && !_draw_dense_pointcloud)
            {
                draw_landmarks_plane();
            }

            // FW: draw landmark line
            if (map_publisher_->using_line_tracking())
            {
                draw_landmarks_line();
            }

            // FW: run frame loop to advance window events
            pangolin::FinishFrame();

            // 2. draw the current frame image

            if (map_publisher_->seg_or_not())
            {
                // FW: concatenate segmentation image for better visualization
                cv::Mat tracked_img = frame_publisher_->draw_frame();
                cv::Mat segmented_img = frame_publisher_->draw_seg_mask();
                cv::Mat multi_imgs;
                vconcat(tracked_img, segmented_img, multi_imgs);
                cv::imshow(frame_viewer_name_, multi_imgs);
            }
            else
            {
                // FW: inside draw_frame(), the tracked keypoints are visualized as cv::rectangle()
                // FW: if using line_tracking, the line segments will be visualized as well
                cv::imshow(frame_viewer_name_, frame_publisher_->draw_frame());
            }

            cv::waitKey(interval_ms_);

            // 3. state transition

            if (*menu_reset_)
            {
                reset();
            }

            check_state_transition();

            // 4. check termination flag

            if (*menu_terminate_ || pangolin::ShouldQuit())
            {
                request_terminate();
            }

            if (terminate_is_requested())
            {
                break;
            }
        }

        if (system_->tracker_is_paused())
        {
            system_->resume_tracker();
        }

        system_->request_terminate();

        terminate();
    }

    void viewer::create_menu_panel()
    {
        pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
        menu_follow_camera_ = std::unique_ptr<pangolin::Var<bool>>(new pangolin::Var<bool>("menu.Follow Camera", true, true));
        menu_grid_ = std::unique_ptr<pangolin::Var<bool>>(new pangolin::Var<bool>("menu.Show Grid", false, true));
        menu_show_keyfrms_ = std::unique_ptr<pangolin::Var<bool>>(new pangolin::Var<bool>("menu.Show Keyframes", true, true));
        menu_show_lms_ = std::unique_ptr<pangolin::Var<bool>>(new pangolin::Var<bool>("menu.Show Landmarks", true, true));
        menu_show_local_map_ = std::unique_ptr<pangolin::Var<bool>>(new pangolin::Var<bool>("menu.Show Local Map", true, true));
        menu_show_graph_ = std::unique_ptr<pangolin::Var<bool>>(new pangolin::Var<bool>("menu.Show Graph", true, true));
        menu_mapping_mode_ = std::unique_ptr<pangolin::Var<bool>>(new pangolin::Var<bool>("menu.Mapping", mapping_mode_, true));
        menu_loop_detection_mode_ = std::unique_ptr<pangolin::Var<bool>>(new pangolin::Var<bool>("menu.Loop Detection", loop_detection_mode_, true));
        menu_pause_ = std::unique_ptr<pangolin::Var<bool>>(new pangolin::Var<bool>("menu.Pause", false, true));
        menu_reset_ = std::unique_ptr<pangolin::Var<bool>>(new pangolin::Var<bool>("menu.Reset", false, false));
        menu_terminate_ = std::unique_ptr<pangolin::Var<bool>>(new pangolin::Var<bool>("menu.Terminate", false, false));
        menu_frm_size_ = std::unique_ptr<pangolin::Var<float>>(new pangolin::Var<float>("menu.Frame Size", 1.0, 1e-1, 1e1, true));
        menu_lm_size_ = std::unique_ptr<pangolin::Var<float>>(new pangolin::Var<float>("menu.Landmark Size", 1.0, 1e-1, 1e1, true));

        if (map_publisher_->seg_or_not())
        {
            _menu_show_plane = std::unique_ptr<pangolin::Var<bool>>(new pangolin::Var<bool>("menu.Show Planes", true, true));
        }

        if (map_publisher_->using_line_tracking())
        {
            _menu_show_lines = std::unique_ptr<pangolin::Var<bool>>(new pangolin::Var<bool>("menu.Show Lines", true, true));
        }
    }

    void viewer::follow_camera(const pangolin::OpenGlMatrix &gl_cam_pose_wc)
    {
        if (*menu_follow_camera_ && follow_camera_)
        {
            s_cam_->Follow(gl_cam_pose_wc);
        }
        else if (*menu_follow_camera_ && !follow_camera_)
        {
            s_cam_->SetModelViewMatrix(pangolin::ModelViewLookAt(viewpoint_x_, viewpoint_y_, viewpoint_z_, 0, 0, 0, 0.0, -1.0, 0.0));
            s_cam_->Follow(gl_cam_pose_wc);
            follow_camera_ = true;
        }
        else if (!*menu_follow_camera_ && follow_camera_)
        {
            follow_camera_ = false;
        }
    }

    void viewer::draw_horizontal_grid()
    {
        if (!*menu_grid_)
        {
            return;
        }

        Eigen::Matrix4f origin;
        origin << 0, 0, 1, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1;
        glPushMatrix();
        glMultTransposeMatrixf(origin.data());

        glLineWidth(1);
        glColor3fv(cs_.grid_.data());

        glBegin(GL_LINES);

        constexpr float interval_ratio = 0.1;
        constexpr float grid_min = -100.0f * interval_ratio;
        constexpr float grid_max = 100.0f * interval_ratio;

        for (int x = -10; x <= 10; x += 1)
        {
            draw_line(x * 10.0f * interval_ratio, grid_min, 0, x * 10.0f * interval_ratio, grid_max, 0);
        }
        for (int y = -10; y <= 10; y += 1)
        {
            draw_line(grid_min, y * 10.0f * interval_ratio, 0, grid_max, y * 10.0f * interval_ratio, 0);
        }

        glEnd();

        glPopMatrix();
    }

    pangolin::OpenGlMatrix viewer::get_current_cam_pose()
    {
        const auto cam_pose_cw = map_publisher_->get_current_cam_pose();
        const pangolin::OpenGlMatrix gl_cam_pose_wc(cam_pose_cw.inverse().eval());
        return gl_cam_pose_wc;
    }

    void viewer::draw_current_cam_pose(const pangolin::OpenGlMatrix &gl_cam_pose_wc)
    {
        // frustum size of the frame
        const float w = camera_size_ * *menu_frm_size_;

        glLineWidth(camera_line_width_);
        glColor3fv(cs_.curr_cam_.data());
        draw_camera(gl_cam_pose_wc, w);
    }

    void viewer::draw_keyframes()
    {
        // frustum size of keyframes
        const float w = keyfrm_size_ * *menu_frm_size_;

        std::vector<PLPSLAM::data::keyframe *> keyfrms;
        map_publisher_->get_keyframes(keyfrms);

        if (*menu_show_keyfrms_)
        {
            glLineWidth(keyfrm_line_width_);
            glColor3fv(cs_.kf_line_.data());
            for (const auto keyfrm : keyfrms)
            {
                if (!keyfrm || keyfrm->will_be_erased())
                {
                    continue;
                }

                draw_camera(keyfrm->get_cam_pose_inv(), w);

                // FW: code for dense reconstruction using depth
                if (_draw_dense_pointcloud)
                {
                    draw_dense_pointcloud(keyfrm);
                }
            }
        }

        if (*menu_show_graph_)
        {
            glLineWidth(graph_line_width_);
            glColor4fv(cs_.graph_line_.data());

            const auto draw_edge = [](const PLPSLAM::Vec3_t &cam_center_1, const PLPSLAM::Vec3_t &cam_center_2)
            {
                glVertex3fv(cam_center_1.cast<float>().eval().data());
                glVertex3fv(cam_center_2.cast<float>().eval().data());
            };

            glBegin(GL_LINES);

            for (const auto keyfrm : keyfrms)
            {
                if (!keyfrm || keyfrm->will_be_erased())
                {
                    continue;
                }

                const PLPSLAM::Vec3_t cam_center_1 = keyfrm->get_cam_center();

                // covisibility graph
                const auto covisibilities = keyfrm->graph_node_->get_covisibilities_over_weight(100);
                if (!covisibilities.empty())
                {
                    for (const auto covisibility : covisibilities)
                    {
                        if (!covisibility || covisibility->will_be_erased())
                        {
                            continue;
                        }
                        if (covisibility->id_ < keyfrm->id_)
                        {
                            continue;
                        }
                        const PLPSLAM::Vec3_t cam_center_2 = covisibility->get_cam_center();
                        draw_edge(cam_center_1, cam_center_2);
                    }
                }

                // spanning tree
                auto spanning_parent = keyfrm->graph_node_->get_spanning_parent();
                if (spanning_parent)
                {
                    const PLPSLAM::Vec3_t cam_center_2 = spanning_parent->get_cam_center();
                    draw_edge(cam_center_1, cam_center_2);
                }

                // loop edges
                const auto loop_edges = keyfrm->graph_node_->get_loop_edges();
                for (const auto loop_edge : loop_edges)
                {
                    if (!loop_edge)
                    {
                        continue;
                    }
                    if (loop_edge->id_ < keyfrm->id_)
                    {
                        continue;
                    }
                    const PLPSLAM::Vec3_t cam_center_2 = loop_edge->get_cam_center();
                    draw_edge(cam_center_1, cam_center_2);
                }
            }

            glEnd();
        }
    }

    void viewer::draw_landmarks()
    {
        if (!*menu_show_lms_)
        {
            return;
        }

        std::vector<PLPSLAM::data::landmark *> landmarks;
        std::set<PLPSLAM::data::landmark *> local_landmarks;

        map_publisher_->get_landmarks(landmarks, local_landmarks);

        if (landmarks.empty())
        {
            return;
        }

        glBegin(GL_POINTS);

        for (const auto lm : landmarks)
        {
            if (!lm || lm->will_be_erased())
            {
                continue;
            }
            if (*menu_show_local_map_ && local_landmarks.count(lm))
            {
                continue;
            }

            glPointSize(point_size_ * *menu_lm_size_);
            glColor3fv(cs_.lm_.data());

            const PLPSLAM::Vec3_t pos_w = lm->get_pos_in_world();
            glVertex3fv(pos_w.cast<float>().eval().data());
        }

        glEnd();

        if (!*menu_show_local_map_)
        {
            return;
        }

        glBegin(GL_POINTS);

        for (const auto local_lm : local_landmarks)
        {
            if (local_lm->will_be_erased())
            {
                continue;
            }

            glPointSize(point_size_ * *menu_lm_size_);
            glColor3fv(cs_.local_lm_.data());

            const PLPSLAM::Vec3_t pos_w = local_lm->get_pos_in_world();
            glVertex3fv(pos_w.cast<float>().eval().data());
        }

        glEnd();
    }

    // FW:
    void viewer::draw_landmarks_plane()
    {
        if (!*_menu_show_plane)
        {
            return;
        }

        std::vector<PLPSLAM::data::Plane *> planes;
        map_publisher_->get_landmark_planes(planes);

        if (planes.empty())
            return;

        auto PlaneColors = map_publisher_->get_available_color();
        auto c = PlaneColors.begin();

        for (auto const &plane : planes)
        {
            if (!plane->is_valid())
            {
                continue;
            }

            auto const map_pts = plane->get_landmarks();

            if (map_pts.empty())
            {
                continue;
            }

            if (c == PlaneColors.end())
            {
                // spdlog::info("Gotta get more colors!?");
                c = PlaneColors.begin();
            }

            auto const &color = *(c++);

            // assign a color to the plane
            if (!plane->_has_color)
            {
                plane->_b = color._b;
                plane->_r = color._r;
                plane->_g = color._g;

                plane->_has_color = true;
            }

            // simply pick two map points to formulate the base vectors of a plane patch
            Eigen::Vector3d baseVector1{0, 0, 0};
            Eigen::Vector3d baseVector2{0, 0, 0};
            for (unsigned int i = 0; i < map_pts.size(); i++)
            {
                if (map_pts[i]->will_be_erased() ||
                    (plane->calculate_distance(map_pts[i]->get_pos_in_world()) > plane->get_best_error()))
                {
                    continue;
                }

                for (unsigned int j = i + 1; j < map_pts.size() - 1; j++)
                {
                    if (map_pts[j]->will_be_erased() ||
                        (plane->calculate_distance(map_pts[j]->get_pos_in_world()) > plane->get_best_error()))
                    {
                        continue;
                    }

                    baseVector1 = (map_pts[i]->get_pos_in_world() - map_pts[j]->get_pos_in_world()).normalized();
                    baseVector2 = baseVector1.cross(plane->get_normal().normalized());

                    break;
                }

                break;
            }

            // visualize with plane-patch
            for (auto const &point : map_pts)
            {
                glBegin(GL_TRIANGLE_FAN);
                glColor4f(plane->_r, plane->_g, plane->_b, _transparency_alpha); // r,g,b, transparency
                if (!point->will_be_erased() && point->get_Owning_Plane())
                {
                    // draw the plane patch centered around the point
                    auto const tr = point->get_pos_in_world() + _square_size * (baseVector1 + baseVector2);
                    auto const br = point->get_pos_in_world() + _square_size * (baseVector1 - baseVector2);
                    auto const bl = point->get_pos_in_world() - _square_size * (baseVector1 + baseVector2);
                    auto const tl = point->get_pos_in_world() + _square_size * (baseVector2 - baseVector1);
                    glVertex3f(tr(0), tr(1), tr(2));
                    glVertex3f(br(0), br(1), br(2));
                    glVertex3f(bl(0), bl(1), bl(2));
                    glVertex3f(tl(0), tl(1), tl(2));
                }
                glEnd();
            }

            // only visualize the normal and base vectors of the dominate plane/stable plane,
            // e.g. new detected plane will not be visualized, this make the visualization more clean
            if (_draw_plane_normal)
            {
                double center_x = 0.0;
                double center_y = 0.0;
                double center_z = 0.0;
                unsigned int num_points = 0;
                for (auto const &point : map_pts)
                {
                    if (!point->will_be_erased() && point->get_Owning_Plane())
                    {

                        center_x += point->get_pos_in_world()(0);
                        center_y += point->get_pos_in_world()(1);
                        center_z += point->get_pos_in_world()(2);
                        num_points++;
                    }
                }

                // update plane centroid information
                center_x = center_x / num_points;
                center_y = center_y / num_points;
                center_z = center_z / num_points;
                Eigen::Vector3d center_point(center_x, center_y, center_z);
                plane->setCentroid(center_point);

                if ((!plane->need_refinement()) && plane->get_num_landmarks() > 36)
                {
                    glEnable(GL_LINE_SMOOTH);
                    glLineWidth(30.0f);

                    glBegin(GL_LINES);
                    // auto const p1 = center_point;
                    auto const p2 = center_point + 4 * _square_size * plane->get_normal().normalized();
                    auto const p3 = center_point + 4 * _square_size * baseVector1;
                    auto const p4 = center_point + 4 * _square_size * baseVector2;

                    // draw the plane normal (color is same as the corresponding plane)
                    glColor3f(plane->_r, plane->_g, plane->_b);
                    glVertex3f(center_point(0), center_point(1), center_point(2));
                    glVertex3f(p2(0), p2(1), p2(2));

                    // draw the plane base_vector_1 (yellow)
                    glColor3f(1.0f, 1.0f, 0.0f);
                    glVertex3f(center_point(0), center_point(1), center_point(2));
                    glVertex3f(p3(0), p3(1), p3(2));

                    // draw the plane base_vector_2 (yellow)
                    glColor3f(1.0f, 1.0f, 0.0f);
                    glVertex3f(center_point(0), center_point(1), center_point(2));
                    glVertex3f(p4(0), p4(1), p4(2));
                    glEnd();
                }
            }
        }
    }

    // FW:
    void viewer::draw_dense_pointcloud(PLPSLAM::data::keyframe *kf)
    {
        // RGB-D dense reconstruction: simply back-project pixel into 3D space
        cv::Mat depth_img = kf->get_depth_map();
        cv::Mat img_rgb = kf->get_img_rgb();
        auto camera_ = kf->camera_;
        auto camera = static_cast<PLPSLAM::camera::perspective *>(camera_);

        glPointSize(point_size_ * *menu_lm_size_);
        glBegin(GL_POINTS);
        for (int r = 0; r < depth_img.rows; r++)
        {
            for (int c = 0; c < depth_img.cols; c++)
            {
                if (0.0 < depth_img.at<float>(r, c))
                {
                    int x = c;
                    int y = r;
                    float unproj_x = ((float)x - camera->cx_) * depth_img.at<float>(r, c) * camera->fx_inv_;
                    float unproj_y = ((float)y - camera->cy_) * depth_img.at<float>(r, c) * camera->fy_inv_;

                    Eigen::Vector3d pos_c{unproj_x, unproj_y, depth_img.at<float>(r, c)};
                    auto rot_wc = kf->get_rotation();
                    auto pos_w = rot_wc.transpose() * pos_c + kf->get_cam_center();

                    cv::Vec3b color = img_rgb.at<cv::Vec3b>(r, c);
                    float red = color.val[0];
                    float green = color.val[1];
                    float blue = color.val[2];
                    std::array<float, 3> color_array = {static_cast<float>(blue / 255.0), static_cast<float>(green / 255.0), static_cast<float>(red / 255.0)};
                    glColor3fv(color_array.data());
                    glVertex3fv(pos_w.cast<float>().eval().data());
                }
            }
        }
        glEnd();
    }

    // FW:
    void viewer::draw_landmarks_line()
    {

        if (!*_menu_show_lines)
        {
            return;
        }

        std::vector<PLPSLAM::data::Line *> landmarks_line;
        map_publisher_->get_landmark_lines(landmarks_line);

        if (landmarks_line.empty())
        {
            return;
        }

        glLineWidth(5.0f);
        glBegin(GL_LINES);
        glColor3f(0.4, 0.35, 0.8);

        for (const auto lm : landmarks_line)
        {
            if (!lm)
                continue;

            PLPSLAM::Vec6_t pos_w = lm->get_pos_in_world();
            glVertex3f(pos_w(0), pos_w(1), pos_w(2));
            glVertex3f(pos_w(3), pos_w(4), pos_w(5));
        }
        glEnd();
    }

    void viewer::draw_camera(const pangolin::OpenGlMatrix &gl_cam_pose_wc, const float width) const
    {
        glPushMatrix();
#ifdef HAVE_GLES
        glMultMatrixf(cam_pose_wc.m);
#else
        glMultMatrixd(gl_cam_pose_wc.m);
#endif

        glBegin(GL_LINES);
        draw_frustum(width);
        glEnd();

        glPopMatrix();
    }

    void viewer::draw_camera(const PLPSLAM::Mat44_t &cam_pose_wc, const float width) const
    {
        glPushMatrix();
        glMultMatrixf(cam_pose_wc.transpose().cast<float>().eval().data());

        glBegin(GL_LINES);
        draw_frustum(width);
        glEnd();

        glPopMatrix();
    }

    void viewer::draw_frustum(const float w) const
    {
        const float h = w * 0.75f;
        const float z = w * 0.6f;
        // 四角錐の斜辺
        draw_line(0.0f, 0.0f, 0.0f, w, h, z);
        draw_line(0.0f, 0.0f, 0.0f, w, -h, z);
        draw_line(0.0f, 0.0f, 0.0f, -w, -h, z);
        draw_line(0.0f, 0.0f, 0.0f, -w, h, z);
        // 四角錐の底辺
        draw_line(w, h, z, w, -h, z);
        draw_line(-w, h, z, -w, -h, z);
        draw_line(-w, h, z, w, h, z);
        draw_line(-w, -h, z, w, -h, z);
    }

    void viewer::reset()
    {
        // reset menu checks
        *menu_follow_camera_ = true;
        *menu_show_keyfrms_ = true;
        *menu_show_lms_ = true;
        *menu_show_local_map_ = true;
        *menu_show_graph_ = true;
        *menu_mapping_mode_ = mapping_mode_;
        *menu_loop_detection_mode_ = loop_detection_mode_;

        // reset menu button
        *menu_reset_ = false;
        *menu_terminate_ = false;

        // reset mapping mode
        if (mapping_mode_)
        {
            system_->enable_mapping_module();
        }
        else
        {
            system_->disable_mapping_module();
        }

        // reset loop detector
        if (loop_detection_mode_)
        {
            system_->enable_loop_detector();
        }
        else
        {
            system_->disable_loop_detector();
        }

        // reset internal state
        follow_camera_ = true;

        // execute reset
        system_->request_reset();
    }

    void viewer::check_state_transition()
    {
        // pause of tracker
        if (*menu_pause_ && !system_->tracker_is_paused())
        {
            system_->pause_tracker();
        }
        else if (!*menu_pause_ && system_->tracker_is_paused())
        {
            system_->resume_tracker();
        }

        // mapping module
        if (*menu_mapping_mode_ && !mapping_mode_)
        {
            system_->enable_mapping_module();
            mapping_mode_ = true;
        }
        else if (!*menu_mapping_mode_ && mapping_mode_)
        {
            system_->disable_mapping_module();
            mapping_mode_ = false;
        }

        // loop detector
        if (*menu_loop_detection_mode_ && !loop_detection_mode_)
        {
            system_->enable_loop_detector();
            loop_detection_mode_ = true;
        }
        else if (!*menu_loop_detection_mode_ && loop_detection_mode_)
        {
            system_->disable_loop_detector();
            loop_detection_mode_ = false;
        }
    }

    void viewer::request_terminate()
    {
        std::lock_guard<std::mutex> lock(mtx_terminate_);
        terminate_is_requested_ = true;
    }

    bool viewer::is_terminated()
    {
        std::lock_guard<std::mutex> lock(mtx_terminate_);
        return is_terminated_;
    }

    bool viewer::terminate_is_requested()
    {
        std::lock_guard<std::mutex> lock(mtx_terminate_);
        return terminate_is_requested_;
    }

    void viewer::terminate()
    {
        std::lock_guard<std::mutex> lock(mtx_terminate_);
        is_terminated_ = true;
    }

    void viewer::load_configuration(const std::string path)
    {
        YAML::Node yaml_node = YAML::LoadFile(path);

        _draw_dense_pointcloud = yaml_node["Threshold.draw_dense_pointcloud"].as<bool>();
        _draw_plane_normal = yaml_node["Threshold.draw_plane_normal"].as<bool>();

        _square_size = yaml_node["Threshold.square_size"].as<double>();
        _transparency_alpha = yaml_node["Threshold.transparency_alpha"].as<float>();
    }

} // namespace pangolin_viewer
