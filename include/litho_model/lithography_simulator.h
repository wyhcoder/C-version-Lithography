#pragma once

#include "grid.h"
#include "pupil.h"
#include "source.h"
#include "mask.h"
#include "simulation_parameters.h"
#include "ep_select.h"


namespace litho {
    class LithographySimulator{
        public:
            LithographySimulator(const SimulationParameters& params);
            // ── 成员（声明顺序 = 构造顺序，_grid 必须在依赖它的成员之前）──
            SimulationParameters _params;
            Grid                 _grid;
            Pupil                _pupil;
            Source               _source;
            Mask                 _mask;
            EpSelect             _ep_select;

        
            // 用 SimulationParameters 组装 Pupil / Source 各自的参数
            // （静态，供成员初始化列表调用；不依赖 this）
            static Params        _make_pupil_params (const SimulationParameters& p);
            static SourceParams  _make_source_params(const SimulationParameters& p, const Eigen::VectorXd& fx);

            
          
    };

}
